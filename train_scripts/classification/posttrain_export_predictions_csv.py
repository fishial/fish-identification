#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference helper for per-image prediction export to CSV (FiftyOne or filepaths).

Outputs one row per image filepath:
sample_id, filepath, ground_truth, prediction, prediction_species_id, score, distance, top5_predictions
"""

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw

# Ensure project root (containing "fish-identification") is on sys.path
CURRENT_FILE = os.path.abspath(__file__)
DELIMITER = "fish-identification"
pos = CURRENT_FILE.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FILE[: pos + len(DELIMITER)])

import fiftyone as fo

from module.classification_package.src.datamodule import ImageEmbeddingDataModule
from module.classification_package.src.lightning_trainer_fixed import (
    ImageEmbeddingTrainerConvnext,
    ImageEmbeddingTrainerViT,
)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v!r}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _infer_image_size_from_backbone_name(backbone_model_name: str) -> int | None:
    for s in (512, 384, 256, 224):
        if backbone_model_name.endswith(f"_{s}"):
            return s
    return None


def _validate_maxvit_resolution(backbone_model_name: str, image_size: int) -> None:
    name = backbone_model_name.lower()
    if ("maxvit" in name or "maxxvit" in name) and image_size is not None:
        inferred = _infer_image_size_from_backbone_name(backbone_model_name)
        if inferred is not None and inferred != image_size:
            raise ValueError(
                f"Incompatible --image_size {image_size} for backbone '{backbone_model_name}'. "
                f"This variant is typically fixed to {inferred}. "
                f"Use --image_size {inferred}, or switch backbone to a matching variant (e.g. *_384) if you want 384."
            )


def _get_sample_field(sample: fo.Sample, field_name: Optional[str]):
    if not field_name:
        return None
    if hasattr(sample, "get_field"):
        try:
            return sample.get_field(field_name)
        except Exception:
            pass
    try:
        return sample[field_name]
    except Exception:
        return None


def _get_polyline(sample: fo.Sample, label_field: str):
    obj = _get_sample_field(sample, label_field)
    if obj is None:
        return None
    return obj


def _polyline_label(polyline) -> Optional[str]:
    if polyline is None:
        return None
    if hasattr(polyline, "label"):
        return polyline.label
    try:
        return polyline.get("label")
    except Exception:
        return None


def _polyline_points(polyline) -> Optional[List[List[float]]]:
    if polyline is None:
        return None
    if hasattr(polyline, "points"):
        return polyline.points
    try:
        return polyline.get("points")
    except Exception:
        return None


def _build_records_from_view(
    dataset_view: fo.DatasetView,
    label_field: str,
) -> Dict[str, List[dict]]:
    records: Dict[str, List[dict]] = {}
    for sample in dataset_view:
        polyline = _get_polyline(sample, label_field)
        label = _polyline_label(polyline)
        points = _polyline_points(polyline)
        if not label or not points or not points[0]:
            continue

        width = _get_sample_field(sample, "width")
        height = _get_sample_field(sample, "height")
        if width is None or height is None:
            continue

        poly = [[int(p[0] * width), int(p[1] * height)] for p in points[0]]

        item = {
            "id": _get_sample_field(sample, "annotation_id"),
            "name": label,
            "base_name": os.path.basename(sample["filepath"]),
            "image_id": _get_sample_field(sample, "image_id"),
            "poly": poly,
            "file_name": sample["filepath"],
            "sample_id": str(sample.id),
        }
        records.setdefault(label, []).append(item)
    return records


def _label_from_value(value) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "label"):
        return value.label
    return str(value)


def resolve_ground_truth(sample, label_field: str = "ground_truth") -> Optional[str]:
    raw_value = None
    try:
        raw_value = sample[label_field]
    except (KeyError, AttributeError, ValueError):
        raw_value = getattr(sample, label_field, None)

    if raw_value is not None:
        return _label_from_value(raw_value)

    fallback_fields = ("polyline", "polylines", "detections", "labels")
    for field in fallback_fields:
        attr = getattr(sample, field, None)
        if attr is None:
            continue
        if hasattr(attr, "label"):
            label = _label_from_value(attr.label)
            if label:
                return label

        if hasattr(attr, "__iter__") and not isinstance(attr, (str, bytes)):
            for item in attr:
                label = _label_from_value(item)
                if label:
                    return label
        else:
            label = _label_from_value(attr)
            if label:
                return label

    return None


def _resolve_polyline_points(
    sample: fo.Sample, label_field: str
) -> Optional[List[List[float]]]:
    polyline = _get_polyline(sample, label_field)
    points = _polyline_points(polyline)
    if not points or not points[0]:
        return None
    width = _get_sample_field(sample, "width")
    height = _get_sample_field(sample, "height")
    if width is None or height is None:
        try:
            with Image.open(sample["filepath"]) as img:
                width, height = img.size
        except Exception:
            return None
    return [[int(p[0] * width), int(p[1] * height)] for p in points[0]]


def pick_top_prediction(predictions: List["Prediction"]) -> Optional["Prediction"]:
    """Return the top prediction (predictions should already be sorted by accuracy descending)."""
    if not predictions:
        return None
    return predictions[0]


def _create_prediction_row(
    sample_id: str,
    filepath: str,
    ground_truth: str,
    top_pred: Optional["Prediction"],
    top5_predictions: str,
) -> dict:
    """Helper to create a standardized prediction row dict."""
    if top_pred is None:
        return {
            "sample_id": sample_id,
            "filepath": filepath,
            "ground_truth": ground_truth,
            "prediction": None,
            "prediction_species_id": None,
            "score": None,
            "distance": None,
            "top5_predictions": top5_predictions,
        }
    return {
        "sample_id": sample_id,
        "filepath": filepath,
        "ground_truth": ground_truth,
        "prediction": top_pred.name,
        "prediction_species_id": top_pred.species_id,
        "score": top_pred.accuracy,
        "distance": top_pred.distance,
        "top5_predictions": top5_predictions,
    }


def load_image_from_view(sample, dataset):
    if hasattr(sample, "load_image"):
        img = sample.load_image()
    else:
        fallback = dataset[sample.id]
        if hasattr(fallback, "load_image"):
            img = fallback.load_image()
        elif getattr(fallback, "filepath", None):
            img = Image.open(fallback.filepath).convert("RGB")
        else:
            raise AttributeError("Sample has no load_image method and no filepath")

    if isinstance(img, Image.Image):
        img = np.array(img)
    elif not isinstance(img, np.ndarray):
        raise TypeError(f"Unexpected image type {type(img)} returned for sample {sample.id}")

    return img

def _read_filepaths_from_args(args) -> List[str]:
    paths: List[str] = []
    if args.filepaths:
        print(f"[READ] Adding {len(args.filepaths)} filepaths from command line")
        paths.extend(args.filepaths)
    if args.filepaths_file:
        print(f"[READ] Reading filepaths from file: {args.filepaths_file}")
        if not os.path.exists(args.filepaths_file):
            raise FileNotFoundError(f"Filepaths file not found: {args.filepaths_file}")
        with open(args.filepaths_file, encoding="utf-8") as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if line:
                    paths.append(line)
                    line_count += 1
        print(f"[READ] Read {line_count} filepaths from file")
    
    if not paths:
        print(f"[READ] No filepaths provided")
        return []
    
    print(f"[READ] Validating {len(paths)} file paths...")
    # Validate paths exist
    validated_paths = []
    skipped = 0
    for p in paths:
        if not p:
            continue
        abs_path = os.path.abspath(p)
        if not os.path.exists(abs_path):
            print(f"[WARNING] Path does not exist, skipping: {abs_path}")
            skipped += 1
            continue
        validated_paths.append(abs_path)
    
    if skipped > 0:
        print(f"[READ] Skipped {skipped} invalid paths")
    print(f"[READ] Validated {len(validated_paths)} valid file paths")
    
    return validated_paths


def _mask_from_polyline(image_size: Tuple[int, int], polyline: Optional[List[List[float]]]) -> np.ndarray:
    """Create a binary mask from a polyline. Returns all-ones mask if polyline is invalid."""
    width, height = image_size
    if not polyline or len(polyline) < 3:
        return np.ones((height, width), dtype=np.uint8)
    
    try:
        # Convert and validate points
        pts = [(int(p[0]), int(p[1])) for p in polyline]
        if len(pts) < 3:
            return np.ones((height, width), dtype=np.uint8)
        
        # Create mask using PIL for efficiency
        mask_img = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_img).polygon(pts, outline=1, fill=1)
        mask = np.array(mask_img, dtype=np.uint8)
        
        # Fallback to all-ones if polygon resulted in empty mask
        if mask.sum() == 0:
            return np.ones((height, width), dtype=np.uint8)
        return mask
    except Exception as e:
        # Silent fallback to full mask on any error
        return np.ones((height, width), dtype=np.uint8)


@dataclass
class Prediction:
    name: str
    species_id: str
    accuracy: float
    distance: float


class ImageEmbeddingInference:
    def __init__(self, args):
        print(f"[INIT] Initializing ImageEmbeddingInference...")
        self.args = args
        self.dataset = None
        if args.dataset_name:
            print(f"[INIT] Loading FiftyOne dataset: {args.dataset_name}")
            self.dataset = fo.load_dataset(args.dataset_name)
            if self.dataset is None:
                raise RuntimeError(f"Dataset {args.dataset_name} not found.")
            print(f"[INIT] Dataset loaded successfully. Total samples: {len(self.dataset)}")

        if args.image_size is None:
            args.image_size = _infer_image_size_from_backbone_name(args.backbone_model_name) or 224
            print(f"[INIT] Image size inferred from backbone: {args.image_size}")
        else:
            print(f"[INIT] Using specified image size: {args.image_size}")
        _validate_maxvit_resolution(args.backbone_model_name, args.image_size)

        print(f"[INIT] Building label mapping...")
        if args.labels_path:
            print(f"[INIT] Loading labels from file: {args.labels_path}")
            with open(args.labels_path, encoding="utf-8") as f:
                id_to_label = json.load(f)
            id_to_label = {int(k): v for k, v in id_to_label.items()}
            label_to_id = {label: idx for idx, label in id_to_label.items()}
            print(f"[INIT] Loaded {len(label_to_id)} labels from file")
        else:
            print(f"[INIT] Building label mapping from dataset...")
            if self.dataset is None:
                raise RuntimeError("Provide --labels_path or --dataset_name to build label mapping.")
            
            # Use filtered view if tags specified, otherwise use entire dataset
            if args.tag:
                print(f"[INIT] Filtering dataset by tags: {args.tag}")
                mapping_view = self.dataset.match_tags(list(args.tag))
                print(f"[INIT] Found {mapping_view.count()} samples with tags {args.tag}")
            else:
                print(f"[INIT] Using entire dataset for label mapping")
                mapping_view = self.dataset
            
            records = _build_records_from_view(mapping_view, args.label_field)
            if not records:
                raise RuntimeError(f"No records found in dataset. Check --dataset_name and --tag.")
            label_to_id = {label: idx for idx, label in enumerate(sorted(records.keys()))}
            id_to_label = {idx: label for label, idx in label_to_id.items()}
            print(f"[INIT] Built label mapping: {len(label_to_id)} unique labels")

        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        # species_id mapping will be built on-the-fly during inference
        self.label_to_species_id = {}
        self.species_id_field = args.species_id_field
        self.label_field = args.label_field

        print(f"[INIT] Initializing data transforms (image_size={args.image_size})...")
        self.transform = ImageEmbeddingDataModule(
            dataset_name=args.dataset_name or "inference",
            batch_size=args.batch_size,
            classes_per_batch=1,
            samples_per_class=1,
            image_size=args.image_size,
            num_workers=args.num_workers,
        ).get_transform(is_train=False)
        print(f"[INIT] Data transforms initialized")

        model_cls = ImageEmbeddingTrainerConvnext if "convnext" in args.backbone_model_name else ImageEmbeddingTrainerViT
        model_type = "ConvNeXt" if "convnext" in args.backbone_model_name else "ViT"
        print(f"[INIT] Initializing {model_type} model...")
        print(f"[INIT] Backbone: {args.backbone_model_name}")
        print(f"[INIT] Embedding dim: {args.embedding_dim}, Num classes: {len(label_to_id)}")
        print(f"[INIT] ArcFace parameters: s={args.arcface_s}, m={args.arcface_m}")
        
        self.model = model_cls(
            num_classes=len(label_to_id),
            embedding_dim=args.embedding_dim,
            backbone_model_name=args.backbone_model_name,
            arcface_s=args.arcface_s,
            arcface_m=args.arcface_m,
            lr=1e-4,
            weight_decay=0.0,
            lr_eta_min=1e-7,
            attention_loss_lambda=0.0,
            load_checkpoint=args.checkpoint,
            output_dir="",
            visualize_attention_map=False,
        )
        print(f"[INIT] Model initialized successfully")

        print(f"[INIT] Configuring device...")
        device_choice = str(getattr(args, "device", "auto")).strip().lower()
        if device_choice == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INIT] Device auto-selected: {self.device}")
        elif device_choice in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device, but CUDA is not available.")
            self.device = torch.device("cuda")
            print(f"[INIT] Using CUDA device")
        elif device_choice == "cpu":
            self.device = torch.device("cpu")
            print(f"[INIT] Using CPU device")
        else:
            raise ValueError("Unsupported --device value. Use 'auto', 'cpu', or 'cuda'.")
        
        self.use_amp = bool(args.precision16) and self.device.type == "cuda"
        if self.use_amp:
            print(f"[INIT] Mixed precision (FP16) enabled")
        else:
            print(f"[INIT] Full precision (FP32) mode")
        
        print(f"[INIT] Moving model to {self.device}...")
        self.model.to(self.device)
        self.model.eval()
        print(f"[INIT] Model ready for inference")

    def _get_species_id_for_label(self, sample, label: str) -> str:
        """Get species_id for a label, building the mapping on-the-fly."""
        if not self.species_id_field:
            return ""
        
        # Check if we already have this mapping
        if label in self.label_to_species_id:
            return self.label_to_species_id[label]
        
        # Build mapping from this sample
        if sample is not None:
            species_id = _get_sample_field(sample, self.species_id_field)
            if species_id is not None:
                self.label_to_species_id[label] = species_id
                return str(species_id)
        
        return ""
    
    def _prepare_tensors(
        self,
        filepath: str,
        polyline: Optional[List[List[float]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pil_img = Image.open(filepath).convert("RGB")
        image_np = np.array(pil_img)
        mask_np = _mask_from_polyline(pil_img.size, polyline)

        transformed = self.transform(image=image_np, mask=mask_np)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return image_tensor, mask_tensor.float()

    @torch.no_grad()
    def predict_filepath(
        self,
        filepath: str,
        polyline: Optional[List[List[float]]] = None,
        topk: int = 5,
    ) -> List[Prediction]:
        image_tensor, mask_tensor = self._prepare_tensors(filepath, polyline=polyline)
        return self._predict_tensors(image_tensor, mask_tensor, topk=topk)

    @torch.no_grad()
    def predict_image(
        self,
        image_np: np.ndarray,
        mask_np: Optional[np.ndarray] = None,
        topk: int = 5,
    ) -> List[Prediction]:
        if mask_np is None:
            mask_np = np.ones(image_np.shape[:2], dtype=np.uint8)
        transformed = self.transform(image=image_np, mask=mask_np)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return self._predict_tensors(image_tensor, mask_tensor, topk=topk)

    def _predict_tensors(
        self,
        image_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        topk: int = 5,
    ) -> List[Prediction]:
        image_tensor = image_tensor.unsqueeze(0).to(self.device, non_blocking=True)
        mask_tensor = mask_tensor.unsqueeze(0).to(self.device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            _, probs, _ = self.model(image_tensor, object_mask=mask_tensor)
        probs = probs.detach().cpu().numpy()[0]
        
        # Use argpartition for faster topk selection
        k = min(max(1, topk), len(probs))
        if k < len(probs):
            idx_partition = np.argpartition(probs, -k)[-k:]
            order = idx_partition[np.argsort(probs[idx_partition])[::-1]]
        else:
            order = np.argsort(probs)[::-1]
        
        preds: List[Prediction] = []
        for idx in order:
            idx_int = int(idx)
            label = self.id_to_label.get(idx_int, "")
            score = float(probs[idx_int])
            preds.append(
                Prediction(
                    name=label,
                    species_id=str(self.label_to_species_id.get(label, "")),
                    accuracy=score,
                    distance=score * float(self.args.distance_scale),
                )
            )
        return preds


class EmbeddingClassifier:
    def __init__(self, inference: ImageEmbeddingInference, topk: int):
        self.inference = inference
        self.topk = topk

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Prediction]:
        return self.inference.predict_image(image, mask_np=mask, topk=self.topk)


def evaluate_dataset(
    classifier: EmbeddingClassifier,
    dataset,
    label_field: str = "ground_truth",
    max_samples: Optional[int] = None,
    use_polyline_mask: bool = True,
):
    view = dataset if max_samples is None else dataset.take(max_samples)

    stats = {
        "total": 0,
        "correct": 0,
        "made_predictions": 0,
        "empty_predictions": 0,
        "missing_ground_truth": 0,
    }
    records = []

    total_samples = view.count()
    print(f"[EVAL] Total samples to process: {total_samples}")
    print(f"[EVAL] Label field: {label_field}")
    print(f"[EVAL] Use polyline mask: {use_polyline_mask}")
    
    for sample in tqdm(view, total=total_samples, desc="Инференс"):
        label = resolve_ground_truth(sample, label_field=label_field)
        if label is None:
            stats["missing_ground_truth"] += 1
            continue

        stats["total"] += 1
        image = load_image_from_view(sample, dataset)
        mask = None
        if use_polyline_mask:
            poly = _resolve_polyline_points(sample, label_field)
            if poly is not None:
                mask = _mask_from_polyline((image.shape[1], image.shape[0]), poly)

        predictions = classifier(image, mask=mask)
        # predictions are already sorted by accuracy (descending)
        top_pred = pick_top_prediction(predictions)
        
        # Update species_id for top prediction using actual sample data (on-the-fly)
        if top_pred is not None:
            species_id = classifier.inference._get_species_id_for_label(sample, label)
            # Update the prediction with the correct species_id from this sample
            top_pred = Prediction(
                name=top_pred.name,
                species_id=species_id,
                accuracy=top_pred.accuracy,
                distance=top_pred.distance,
            )
        
        top5_names = [pred.name for pred in predictions[:5] if pred.name]
        top5_string = ";".join(top5_names)

        if top_pred is None:
            stats["empty_predictions"] += 1
        else:
            stats["made_predictions"] += 1
            if top_pred.name == label:
                stats["correct"] += 1

        row = _create_prediction_row(
            sample_id=sample.id,
            filepath=sample.filepath,
            ground_truth=label,
            top_pred=top_pred,
            top5_predictions=top5_string,
        )
        records.append(row)

    stats["accuracy"] = stats["correct"] / stats["made_predictions"] if stats["made_predictions"] else 0.0
    
    print(f"\n[EVAL] Evaluation completed:")
    print(f"[EVAL]   Total samples: {stats['total']}")
    print(f"[EVAL]   Made predictions: {stats['made_predictions']}")
    print(f"[EVAL]   Empty predictions: {stats['empty_predictions']}")
    print(f"[EVAL]   Missing ground truth: {stats['missing_ground_truth']}")
    print(f"[EVAL]   Correct predictions: {stats['correct']}")
    print(f"[EVAL]   Accuracy: {stats['accuracy']:.2%}")
    
    # Show species_id mapping stats if enabled
    if classifier.inference.species_id_field:
        print(f"[EVAL]   Species ID mappings built on-the-fly: {len(classifier.inference.label_to_species_id)}")
    
    return stats, pd.DataFrame(records)


def get_args():
    p = argparse.ArgumentParser(description="Export per-sample predictions to CSV.")
    p.add_argument("--dataset_name", type=str, default="", help="FiftyOne dataset name (optional).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .ckpt file.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write CSV output.")
    p.add_argument(
        "--filepaths",
        type=str,
        nargs="*",
        default=[],
        help="One or more image paths to run inference on.",
    )
    p.add_argument(
        "--filepaths_file",
        type=str,
        default="",
        help="Text file with one image path per line.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples from dataset.")
    p.add_argument("--topk", type=int, default=5, help="Number of top predictions to store.")

    p.add_argument("--image_size", type=int, default=None, help="Input image size for eval.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation dataloader.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--labels_path",
        type=str,
        default="",
        help="Optional path to JSON file with id_to_label produced by the training run.",
    )

    p.add_argument("--backbone_model_name", type=str, default="convnext_small")
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--arcface_s", type=float, default=64.0)
    p.add_argument("--arcface_m", type=float, default=0.2)
    p.add_argument("--precision16", type=_str2bool, default=True, help="Use autocast fp16 on CUDA (true/false).")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on: auto, cpu, or cuda.",
    )

    p.add_argument("--label_field", type=str, default="polyline", help="FiftyOne field for polyline label.")
    p.add_argument(
        "--tag",
        type=str,
        nargs="*",
        default=[],
        help="Optional dataset tag(s) to filter samples (e.g. 'train', 'val', 'test'). If not specified, uses entire dataset.",
    )
    p.add_argument(
        "--species_id_field",
        type=str,
        default="",
        help="Optional FiftyOne field name to map label -> species_id (e.g. 'species_id' or 'drawn_fish_id').",
    )
    p.add_argument(
        "--distance_scale",
        type=float,
        default=10.0,
        help="Distance = score * distance_scale (set to 0 to disable scaling).",
    )
    p.add_argument(
        "--use_polyline_mask",
        type=_str2bool,
        default=True,
        help="Use polyline mask for inference on dataset samples (true/false).",
    )
    return p.parse_args()


def main():
    print("="*80)
    print("[MAIN] Starting prediction export script")
    print("="*80)
    
    print("[MAIN] Parsing command line arguments...")
    args = get_args()
    print(f"[MAIN] Arguments parsed successfully")
    
    print(f"[MAIN] Validating checkpoint path: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    print(f"[MAIN] Checkpoint file exists")

    # Validate we have input data
    print(f"[MAIN] Reading input filepaths...")
    filepaths = _read_filepaths_from_args(args)
    if filepaths:
        print(f"[MAIN] Found {len(filepaths)} filepaths to process")
    else:
        print(f"[MAIN] No filepaths provided, will use dataset")
    
    if not filepaths and not args.dataset_name:
        raise ValueError("Must provide either --dataset_name or --filepaths/--filepaths_file")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"predictions_{timestamp}")
    print(f"[MAIN] Creating output directory: {out_dir}")
    _ensure_dir(out_dir)
    print(f"[MAIN] Output directory created")

    print("\n" + "="*80)
    print("[MAIN] Initializing inference engine...")
    print("="*80)
    inference = ImageEmbeddingInference(args)
    print("[MAIN] Inference engine initialized successfully")
    print("="*80 + "\n")
    rows = []
    stats = {
        "total": 0,
        "correct": 0,
        "made_predictions": 0,
        "empty_predictions": 0,
        "missing_ground_truth": 0,
    }

    if filepaths:
        print(f"[INFERENCE] Processing {len(filepaths)} image files...")
        print(f"[INFERENCE] Creating classifier (topk={args.topk})...")
        classifier = EmbeddingClassifier(inference, topk=args.topk)
        print(f"[INFERENCE] Starting batch inference...")
        for path in tqdm(filepaths, desc="Processing filepaths"):
            try:
                image = np.array(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"[WARNING] Failed to load {path}: {e}")
                row = _create_prediction_row("", path, "", None, "")
                rows.append(row)
                stats["empty_predictions"] += 1
                continue
                
            predictions = classifier(image, mask=None)
            # predictions are already sorted by accuracy (descending)
            top_pred = pick_top_prediction(predictions)
            top5_names = [pred.name for pred in predictions[:args.topk] if pred.name]
            top5_string = ";".join(top5_names)

            if top_pred is None:
                stats["empty_predictions"] += 1
            else:
                stats["made_predictions"] += 1

            row = _create_prediction_row("", path, "", top_pred, top5_string)
            rows.append(row)
        
        print(f"\n[INFERENCE] Filepath processing completed:")
        print(f"[INFERENCE]   Total files processed: {len(filepaths)}")
        print(f"[INFERENCE]   Successful predictions: {stats['made_predictions']}")
        print(f"[INFERENCE]   Failed/empty predictions: {stats['empty_predictions']}")
    else:
        print(f"[INFERENCE] Processing FiftyOne dataset...")
        if inference.dataset is None:
            raise RuntimeError("Provide --dataset_name or --filepaths/--filepaths_file.")
        dataset = inference.dataset
        print(f"[INFERENCE] Dataset: {args.dataset_name}, Total samples: {len(dataset)}")
        
        if args.tag:
            print(f"[INFERENCE] Filtering by tags: {args.tag}")
            dataset = dataset.match_tags(list(args.tag))
            print(f"[INFERENCE] Samples after filtering: {dataset.count()}")
        else:
            print(f"[INFERENCE] Using entire dataset (no tag filter)")
        
        if args.max_samples:
            print(f"[INFERENCE] Limiting to max {args.max_samples} samples")
        
        print(f"[INFERENCE] Using polyline mask: {args.use_polyline_mask}")
        print(f"[INFERENCE] Creating classifier (topk={args.topk})...")
        classifier = EmbeddingClassifier(inference, topk=args.topk)
        
        print(f"[INFERENCE] Starting dataset evaluation...")
        stats, df = evaluate_dataset(
            classifier,
            dataset,
            label_field=args.label_field,
            max_samples=args.max_samples,
            use_polyline_mask=bool(args.use_polyline_mask),
        )
        rows = df.to_dict(orient="records")
        print(f"[INFERENCE] Dataset evaluation completed")

    print("\n" + "="*80)
    print("[SAVE] Saving results...")
    print("="*80)
    
    csv_path = os.path.join(out_dir, "predictions.csv")
    print(f"[SAVE] Creating DataFrame with {len(rows)} rows...")
    df = pd.DataFrame(rows)
    
    if args.species_id_field and "prediction_species_id" not in df.columns:
        print(f"[SAVE] Adding empty prediction_species_id column")
        df.insert(4, "prediction_species_id", "")
    
    print(f"[SAVE] Writing CSV to: {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] CSV saved successfully")

    meta_path = os.path.join(out_dir, "meta.json")
    print(f"[SAVE] Writing metadata to: {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": os.path.abspath(args.checkpoint),
                "dataset_name": args.dataset_name,
                "backbone_model_name": args.backbone_model_name,
                "embedding_dim": args.embedding_dim,
                "arcface_s": args.arcface_s,
                "arcface_m": args.arcface_m,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "label_field": args.label_field,
                "tag": args.tag,
                "species_id_field": args.species_id_field,
                "distance_scale": args.distance_scale,
                "device": str(inference.device),
                "amp_fp16": bool(inference.use_amp),
                "n_samples": int(len(rows)),
                "stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[SAVE] Metadata saved successfully")

    print("\n" + "="*80)
    print("[RESULTS] Export completed successfully!")
    print("="*80)
    print(f"[RESULTS] Output directory: {out_dir}")
    print(f"[RESULTS] CSV file: {csv_path}")
    print(f"[RESULTS] Total rows exported: {len(rows)}")
    
    if stats["made_predictions"] > 0:
        print(f"\n[STATS] Inference Statistics:")
        print(f"[STATS]   Total samples processed: {stats.get('total', len(rows))}")
        print(f"[STATS]   Successful predictions: {stats['made_predictions']}")
        print(f"[STATS]   Empty predictions: {stats['empty_predictions']}")
        if stats.get('missing_ground_truth', 0) > 0:
            print(f"[STATS]   Missing ground truth: {stats['missing_ground_truth']}")
        if stats.get('correct', 0) > 0 and stats.get('accuracy') is not None:
            print(f"[STATS]   Correct predictions: {stats['correct']}")
            print(f"[STATS]   Accuracy: {stats['accuracy']:.2%}")
    
    print("="*80)


if __name__ == "__main__":
    print("Start!")
    main()
