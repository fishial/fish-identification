import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from module.classification_package.src.datamodule import ImageEmbeddingDataModule
from module.classification_package.src.lightning_trainer_fixed import (
    ImageEmbeddingTrainerConvnext,
    ImageEmbeddingTrainerViT,
)


logger = logging.getLogger("EmbeddingClassifierV2")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class PredictionResult:
    name: str
    species_id: Optional[Union[str, int]]
    distance: float
    accuracy: float
    image_id: Optional[int]
    annotation_id: Optional[int]
    drawn_fish_id: Optional[int]


def _infer_image_size_from_backbone_name(backbone_model_name: str) -> Optional[int]:
    for s in (512, 384, 256, 224):
        if backbone_model_name.endswith(f"_{s}"):
            return s
    return None


class EmbeddingClassifierV2:
    """
    Inference helper for the updated Lightning-based embedding models.

    Expected config keys (all optional unless noted):
      - model: {
          "checkpoint": <path>,           # required
          "backbone_model_name": <str>,   # required
          "embedding_dim": <int>,         # required
          "arcface_s": <float>,
          "arcface_m": <float>,
          "image_size": <int>,
          "device": "auto|cpu|cuda",
          "precision16": <bool>
        }
      - labels_path: <path to id->label json> (optional if id_to_label provided)
      - id_to_label: {int: str} (optional)
      - label_to_species_id: {label: species_id} (optional)
      - dataset: {"path": <embeddings db path>} (optional, for neighbor search)
      - distance_scale: <float> (default: 10.0)
    """

    def __init__(self, config: Dict):
        logger.setLevel(getattr(logging, config.get("log_level", "INFO").upper()))
        self.config = config

        model_cfg = config.get("model", {})
        checkpoint_path = model_cfg.get("checkpoint")
        if not checkpoint_path:
            raise ValueError("config['model']['checkpoint'] is required")

        backbone_model_name = model_cfg.get("backbone_model_name")
        if not backbone_model_name:
            raise ValueError("config['model']['backbone_model_name'] is required")

        embedding_dim = int(model_cfg.get("embedding_dim", 512))
        arcface_s = float(model_cfg.get("arcface_s", 64.0))
        arcface_m = float(model_cfg.get("arcface_m", 0.2))

        device_choice = str(model_cfg.get("device", "auto")).lower().strip()
        if device_choice == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choice in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device, but CUDA is not available.")
            self.device = torch.device("cuda")
        elif device_choice == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError("Unsupported device option. Use 'auto', 'cpu', or 'cuda'.")

        self.use_amp = bool(model_cfg.get("precision16", True)) and self.device.type == "cuda"

        # Labels
        labels_path = config.get("labels_path") or model_cfg.get("labels_path")
        if labels_path:
            with open(labels_path, encoding="utf-8") as f:
                id_to_label = json.load(f)
            self.id_to_label = {int(k): v for k, v in id_to_label.items()}
        else:
            id_to_label = config.get("id_to_label")
            if not id_to_label:
                raise ValueError("Provide labels_path or id_to_label for inference.")
            self.id_to_label = {int(k): v for k, v in id_to_label.items()}

        self.label_to_species_id = config.get("label_to_species_id", {})

        # Optional embedding database for neighbor search
        self.db_embeddings = None
        self.db_labels = None
        self.image_ids = None
        self.annotation_ids = None
        self.drawn_fish_ids = None
        self.keys = None
        dataset_cfg = config.get("dataset", {})
        if dataset_cfg.get("path"):
            self._load_data(dataset_cfg["path"])
            self._prepare_centroids()

        # Transform (consistent with training)
        image_size = model_cfg.get("image_size")
        if image_size is None:
            image_size = _infer_image_size_from_backbone_name(backbone_model_name) or 224

        self.transform = ImageEmbeddingDataModule(
            dataset_name="inference",
            batch_size=1,
            classes_per_batch=1,
            samples_per_class=1,
            image_size=int(image_size),
            num_workers=0,
        ).get_transform(is_train=False)

        # Load model
        model_cls = ImageEmbeddingTrainerConvnext if "convnext" in backbone_model_name else ImageEmbeddingTrainerViT
        self.model = model_cls(
            num_classes=len(self.id_to_label),
            embedding_dim=embedding_dim,
            backbone_model_name=backbone_model_name,
            arcface_s=arcface_s,
            arcface_m=arcface_m,
            lr=1e-4,
            weight_decay=0.0,
            lr_eta_min=1e-7,
            attention_loss_lambda=0.0,
            load_checkpoint=checkpoint_path,
            output_dir="",
            visualize_attention_map=False,
        )

        self.model.to(self.device)
        self.model.eval()
        self.distance_scale = float(config.get("distance_scale", 10.0))

        logger.info("EmbeddingClassifierV2 initialized on %s", self.device)

    def _label_name(self, label) -> str:
        if isinstance(label, (int, np.integer)):
            return self.id_to_label.get(int(label), str(label))
        return str(label)

    def _load_data(self, dataset_path: str):
        data = torch.load(dataset_path, map_location="cpu")
        embeddings = data["embeddings"]
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = embeddings.astype("float32")
        embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        self.db_embeddings = embeddings
        self.db_labels = np.array(data["labels"])
        self.image_ids = data.get("image_id")
        self.annotation_ids = data.get("annotation_id")
        self.drawn_fish_ids = data.get("drawn_fish_id")
        self.keys = data.get("labels_keys")

        logger.info("Embedding database loaded from %s", dataset_path)

    def __call__(self, img: Union[np.ndarray, List[np.ndarray]], mask: Optional[np.ndarray] = None):
        if isinstance(img, np.ndarray):
            return self.inference_numpy(img, mask=mask)
        if isinstance(img, list) and all(isinstance(i, np.ndarray) for i in img):
            return self.inference_numpy_batch(img, mask=mask)
        raise TypeError("Input must be np.ndarray or List[np.ndarray].")

    def _prepare_tensors(
        self,
        image_np: np.ndarray,
        mask_np: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        if mask_np is None:
            mask_np = np.ones(image_np.shape[:2], dtype=np.uint8)
        transformed = self.transform(image=image_np, mask=mask_np)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return {"image": image_tensor, "mask": mask_tensor.float()}

    def inference_numpy(self, img: np.ndarray, mask: Optional[np.ndarray] = None):
        prepared = self._prepare_tensors(img, mask_np=mask)
        tensor = prepared["image"].unsqueeze(0).to(self.device)
        mask_tensor = prepared["mask"].unsqueeze(0).to(self.device)
        return self._inference_batch_tensor(tensor, mask_tensor)[0]

    def inference_numpy_batch(self, imgs: List[np.ndarray], mask: Optional[np.ndarray] = None):
        batch_imgs = []
        batch_masks = []
        for img in imgs:
            prepared = self._prepare_tensors(img, mask_np=mask)
            batch_imgs.append(prepared["image"])
            batch_masks.append(prepared["mask"])
        tensors = torch.stack(batch_imgs).to(self.device)
        masks = torch.stack(batch_masks).to(self.device)
        return self._inference_batch_tensor(tensors, masks)

    def _inference_batch_tensor(self, tensors: torch.Tensor, masks: torch.Tensor):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                embeddings, probs, _ = self.model(tensors, object_mask=masks)

        top_probabilities, top_indices = torch.topk(probs, 5)
        output = []
        for i, arc_idx in enumerate(top_indices):
            item = {}
            for num_pred, pred_by_arc_head_id in enumerate(arc_idx):
                arc_label = self.id_to_label.get(pred_by_arc_head_id.item(), str(pred_by_arc_head_id.item()))
                score = top_probabilities[i][num_pred]
                if score > 0.1:
                    item[arc_label] = {
                        "index": None,
                        "similarity": round(float(score.item()), 3),
                        "times": 1,
                    }
            output.append(item)

        logger.debug("Inference output (logits-only): %s", output)
        return self._postprocess(output)

    def _postprocess(self, class_results) -> List[List[PredictionResult]]:
        results = []
        for single_fish in class_results:
            fish_results = []
            for label, data in single_fish.items():
                index = data["index"]
                label_name = self._label_name(label)
                fish_results.append(
                    PredictionResult(
                        name=label_name,
                        species_id=self.label_to_species_id.get(label_name),
                        distance=float(data["similarity"]) * self.distance_scale,
                        accuracy=float(data["similarity"]) / max(float(data["times"]), 1.0),
                        image_id=self.image_ids[index] if index is not None and self.image_ids is not None else None,
                        annotation_id=self.annotation_ids[index] if index is not None and self.annotation_ids is not None else None,
                        drawn_fish_id=self.drawn_fish_ids[index] if index is not None and self.drawn_fish_ids is not None else None,
                    )
                )
            results.append(fish_results)
        return results

    def _prepare_centroids(self):
        if self.db_embeddings is None or self.db_labels is None:
            return
        unique_labels = np.unique(self.db_labels)
        self.label_to_centroid = {}
        for label in unique_labels:
            class_embs = self.db_embeddings[self.db_labels == label]
            centroid = np.mean(class_embs, axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-10)
            self.label_to_centroid[label] = centroid

        self.centroid_matrix = np.stack([self.label_to_centroid[label] for label in self.label_to_centroid])
        self.centroid_labels = list(self.label_to_centroid.keys())

    def get_top_neighbors_from_embeddings(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        topk_centroid: int = 5,
        topk_neighbors: int = 10,
        centroid_threshold: float = 0.7,
        neighbor_threshold: float = 0.8,
    ) -> List[Dict[str, Dict[str, Union[float, int, None]]]]:
        if self.db_embeddings is None or self.db_labels is None:
            raise RuntimeError("Embedding database is not loaded. Provide dataset.path in config.")

        start_time = time.time()
        logger.info("Starting search over %d embeddings", len(query_embeddings))

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy().astype("float32")
        query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-10)

        results = []
        for query_emb in query_embeddings:
            centroid_sims = np.dot(self.centroid_matrix, query_emb)
            top_centroid_indices = np.argsort(-centroid_sims)[:topk_centroid]

            centroid_scores = {
                self.centroid_labels[idx]: float(centroid_sims[idx])
                for idx in top_centroid_indices
                if centroid_sims[idx] >= centroid_threshold
            }
            selected_classes = set(centroid_scores.keys())

            if not selected_classes:
                results.append({})
                continue

            class_mask = np.isin(self.db_labels, list(selected_classes))
            selected_embeddings = self.db_embeddings[class_mask]
            selected_labels = self.db_labels[class_mask]
            selected_indices = np.where(class_mask)[0]

            if len(selected_embeddings) == 0:
                results.append({"top_neighbors": [], "centroid_scores": centroid_scores})
                continue

            try:
                import faiss  # type: ignore
                faiss_index = faiss.IndexFlatIP(selected_embeddings.shape[1])
                faiss_index.add(selected_embeddings)
                distances, indices = faiss_index.search(
                    query_emb.reshape(1, -1),
                    min(topk_neighbors, len(selected_embeddings)),
                )
                distances = distances[0]
                indices = indices[0]
            except Exception:
                sims = np.dot(selected_embeddings, query_emb)
                order = np.argsort(-sims)[: min(topk_neighbors, len(selected_embeddings))]
                distances = sims[order]
                indices = order

            score_map: Dict[Union[int, str], Dict[str, Union[float, int, None]]] = {}
            for rank, idx in enumerate(indices):
                label = selected_labels[idx]
                sim = float(distances[rank])
                original_idx = int(selected_indices[idx])
                if sim >= neighbor_threshold and sim != 1.0:
                    if label not in score_map:
                        score_map[label] = {"index": None, "similarity": 0.0, "times": 0}
                    score_map[label]["similarity"] += sim
                    score_map[label]["times"] += 1
                    if score_map[label]["index"] is None:
                        score_map[label]["index"] = original_idx

            for label, sim in centroid_scores.items():
                if label not in score_map:
                    score_map[label] = {"index": None, "similarity": float(sim), "times": 1}

            results.append(score_map)

        logger.info("Completed in %.2f seconds", time.time() - start_time)
        return results
