# -*- coding: utf-8 -*-
"""
Lightweight Inference Engine for TorchScript-exported models.

This module provides a robust, production-ready inference pipeline for fish species 
classification and embedding extraction. It supports standard inference, diagnostic 
debugging, and Test-Time Augmentation (TTA) for high-accuracy predictions.

Prediction pipeline
---------------------------
Supported ``method`` values in ``predict()``:

+----------------------+----------------------------------------------------------+
| method               | score formula                                            |
+======================+==========================================================+
| ``arcface_logits``   | softmax probability from ArcFace head                    |
+----------------------+----------------------------------------------------------+
| ``arcface_centroid`` | MAX cosine similarity to Sub-center ArcFace weights      |
+----------------------+----------------------------------------------------------+
| ``natural_centroid`` | MAX cosine similarity to loaded Natural Centroids (.pt)  |
+----------------------+----------------------------------------------------------+

Use ``predict_all_methods()`` to obtain all three heads after a single forward pass.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Adjust import according to your actual project structure

logger = logging.getLogger(__name__)

__all__ = [
    "InferenceConfig",
    "PredictionResult",
    "FishResult",
    "AllMethodsFishResult",
    "FishInferenceEngine",
    "build_transform",
    "build_tta_transform",
]

ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
MethodType = Literal["arcface_logits", "arcface_centroid", "natural_centroid"]

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """
    Configuration for the inference engine.
    
    Attributes:
        max_unique_classes (int): The maximum number of top predictions to return.
        return_emb (bool): Whether to include the raw embedding vector in the result.
        k_centers (int): Number of sub-centers per class for ArcFace scoring.
    """
    max_unique_classes: int = 5
    return_emb: bool = False
    k_centers: int = 3

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Represents a single class prediction."""
    name: str
    species_id: Optional[int]
    distance: float
    accuracy: float

    @property
    def average_similarity(self) -> float:
        """Alias for accuracy (cosine similarity or softmax prob)."""
        return self.accuracy

@dataclass
class FishResult:
    """Encapsulates the full output of an inference run for a single image."""
    top_k: List[PredictionResult] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    debug_image: Optional[np.ndarray] = None

    @property
    def best(self) -> Optional[PredictionResult]:
        """Returns the highest scoring prediction, if available."""
        return self.top_k[0] if self.top_k else None

@dataclass
class AllMethodsFishResult:
    """Top-k predictions for one image using all scoring heads in a single forward pass."""
    arcface_logits: FishResult
    arcface_centroid: Optional[FishResult]
    natural_centroid: FishResult

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def build_transform(input_size: tuple = (154, 434)) -> A.Compose:
    """Builds the standard validation/inference transform pipeline."""
    logger.debug(f"Building standard transform pipeline with input_size={input_size}")
    return A.Compose([
            A.Resize(height=input_size[0], width=input_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

@lru_cache(maxsize=8)
def build_tta_transform(input_size: tuple = (154, 434)) -> A.Compose:
    """
    Transform pipeline for Test-Time Augmentation with safe photometric augmentations.

    Spatial distortions are intentionally excluded to preserve fish geometry.
    
    Args:
        input_size (tuple): Target resolution (H, W) for the model.
        
    Returns:
        A.Compose: Albumentations pipeline.
    """
    logger.debug(f"Building TTA transform pipeline with input_size={input_size}")
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1], interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def _to_numpy_rgb(image: ImageLike) -> np.ndarray:
    """Converts various image types to a standard RGB NumPy array."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = ((arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8))
        if arr.ndim == 2: # Grayscale to RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[2] == 4: # RGBA to RGB
            arr = arr[:, :, :3]
        return arr
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, torch.Tensor):
        arr = image.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4): # CHW to HWC
            arr = arr.transpose(1, 2, 0)
        return _to_numpy_rgb(arr)
    
    logger.error(f"Failed to convert image of type {type(image)} to NumPy RGB.")
    raise TypeError(f"Unsupported image type: {type(image)}")

# ---------------------------------------------------------------------------
# Core inference engine
# ---------------------------------------------------------------------------

class FishInferenceEngine:
    """
    Main engine for preprocessing, model inference, and post-processing.
    Handles affine transformations, batching, and multiple scoring mechanisms.
    """
    def __init__(
        self,
        model,
        natural_gallery_path: str | Path,
        input_size: tuple = (154, 434), # (H, W)
        device: str = "cpu",
        config: Optional[InferenceConfig] = None,
    ) -> None:
        logger.info(f"Initializing FishInferenceEngine on device '{device}' with input_size={input_size}")
        self.model = model.to(device).eval()
        self.device = device
        self.input_size = input_size
        self.config = config or InferenceConfig()
        self.transform = build_transform(input_size=self.input_size)
        self.tta_transform = build_tta_transform(input_size=self.input_size)

        logger.info(f"Loading natural centroids and class mapping from: {natural_gallery_path}")
        try:
            nat_data = torch.load(natural_gallery_path, map_location=device, weights_only=False)
            logger.debug("Successfully loaded natural gallery file.")
        except Exception as e:
            logger.error(f"Failed to load natural gallery from {natural_gallery_path}: {e}")
            raise

        raw_keys = nat_data.get('labels_keys', {})
        self.class_mapping: Dict[int, dict] = {int(k): v for k, v in raw_keys.items()}
        
        if not self.class_mapping:
            logger.warning("No 'labels_keys' found in the natural gallery file. Predictions will use raw class IDs.")
        else:
            logger.debug(f"Loaded mapping for {len(self.class_mapping)} classes.")

        with torch.no_grad():
            logger.debug("Extracting ArcFace and Natural centroids...")
            # ArcFace centroids from the model's own head weights
            self._arcface_centroids = None

            try:
                if hasattr(self.model, "arcface_head"):
                    self._arcface_centroids = F.normalize(
                        self.model.arcface_head.weight, p=2, dim=1
                    )
                    logger.info("ArcFace centroids loaded from model.")
                else:
                    logger.warning("ArcFace head not found in TorchScript model — arcface_centroid disabled.")
            except Exception as e:
                logger.warning(f"Failed to extract ArcFace centroids: {e}")
                self._arcface_centroids = None

            # Natural centroids from the external .pt file
            raw_cents = nat_data.get('centroids', nat_data.get('embeddings'))
            if raw_cents is None:
                raise KeyError(
                    "Natural gallery must contain 'centroids' or 'embeddings' tensor/array."
                )
            if isinstance(raw_cents, np.ndarray):
                raw_cents = torch.from_numpy(raw_cents)

            if 'labels' not in nat_data:
                raise KeyError("Natural gallery must contain 'labels' aligned with centroids.")
            raw_labels = nat_data['labels']
            if isinstance(raw_labels, np.ndarray):
                raw_labels = torch.from_numpy(raw_labels)

            self._natural_centroids = raw_cents.to(device).float()
            self._natural_labels = raw_labels.to(device).long()
            logger.info(f"Engine initialized. Centroids loaded: {self._natural_centroids.shape[0]}")

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        input_size: tuple = (154, 434),
        device: str = "cpu",
        config: Optional[InferenceConfig] = None,
    ) -> "FishInferenceEngine":
        """
        Load fully self-contained TorchScript bundle.

        Bundle must contain:
            - model (forward -> emb, logits)
            - natural_centroids
            - natural_labels
            - class_mapping_json (optional)
        """
        logger.info(f"Loading bundle from: {bundle_path}")

        try:
            model = torch.jit.load(bundle_path, map_location=device)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load bundle: {e}")
            raise

        engine = cls.__new__(cls)  # bypass __init__

        # --- basic setup ---
        engine.model = model.to(device).eval()
        engine.device = device
        engine.input_size = input_size
        engine.config = config or InferenceConfig()
        engine.transform = build_transform(input_size=input_size)
        engine.tta_transform = build_tta_transform(input_size=input_size)

        # ------------------------------------------------------------------
        # LOAD embedded centroids
        # ------------------------------------------------------------------
        try:
            engine._natural_centroids = model.natural_centroids.to(device).float()
            engine._natural_labels = model.natural_labels.to(device).long()
            logger.info(f"Loaded embedded centroids: {engine._natural_centroids.shape}")
        except AttributeError:
            raise RuntimeError(
                "Bundle does not contain natural centroids. "
                "Export model with embedded centroids."
            )

        # ------------------------------------------------------------------
        # LOAD class mapping (optional)
        # ------------------------------------------------------------------
        engine.class_mapping = {}

        try:
            import json

            if hasattr(model, "class_mapping_json_bytes"):
                json_bytes = bytes(model.class_mapping_json_bytes.cpu().tolist())
                json_str = json_bytes.decode("utf-8")

                engine.class_mapping = {
                    int(k): v for k, v in json.loads(json_str).items()
                }
                
                logger.info(f"Loaded class mapping: {len(engine.class_mapping)} classes")
            else:
                logger.warning("No class_mapping_json found in bundle.")
                engine.class_mapping = {}
        except Exception:
            logger.warning("No class_mapping_json found in bundle.")

        # ------------------------------------------------------------------
        # ArcFace (optional)
        # ------------------------------------------------------------------
        engine._arcface_centroids = None
        try:
            if hasattr(model, "arcface_centroids"):
                engine._arcface_centroids = F.normalize(
                    model.arcface_centroids.to(device), p=2, dim=1
                )
                logger.info("Loaded embedded ArcFace centroids.")
        except Exception as e:
            logger.warning(f"ArcFace centroids not available: {e}")

        return engine

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        natural_gallery_path: str | Path,
        input_size: tuple,
        device: str = "cpu",
        config: Optional[InferenceConfig] = None,
    ) -> "FishInferenceEngine":
        """Instantiates the engine directly from a TorchScript checkpoint."""
        logger.info(f"Loading TorchScript model from: {checkpoint_path}")
        try:
            model = torch.jit.load(checkpoint_path, map_location=device)
            model.eval()
            logger.debug("Model checkpoint loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise

        return cls(
            model=model,
            natural_gallery_path=natural_gallery_path,
            input_size=input_size,
            device=device,
            config=config,
        )

    # ------------------------------------------------------------------
    # Horizontal crop helpers
    # ------------------------------------------------------------------

    def _crop_fallback(
        self,
        image: np.ndarray,
        bbox: Optional[list],
    ) -> tuple[np.ndarray, list]:
        """Fallback crop if bbox is not provided or is invalid."""
        if not bbox or len(bbox) != 4:
            return image, []
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2].copy()
        
        if crop.size == 0:
            return image, []
        return crop, []

    def _crop_horizontal(
        self,
        image: np.ndarray,
        bbox: Optional[list],
        poly: Optional[list],
        debug_mode: bool = False,
    ) -> tuple[np.ndarray, list]:

        if not poly or len(poly) < 3:
            return self._crop_fallback(image, bbox)

        try:
            NEUTRAL_BG = (124, 116, 104)

            poly_np = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)

            # === 1. minAreaRect (стабильная ось) ===
            rect = cv2.minAreaRect(poly_np)
            (cx, cy), _, _ = rect
            box = cv2.boxPoints(rect)

            d01 = np.linalg.norm(box[0] - box[1])
            d12 = np.linalg.norm(box[1] - box[2])

            if d01 > d12:
                dx, dy = box[1] - box[0]
            else:
                dx, dy = box[2] - box[1]

            # фиксируем направление (убираем 180° флип)
            if dx < 0:
                dx, dy = -dx, -dy

            base_angle = math.degrees(math.atan2(dy, dx))

            # === 2. ВРАЩЕНИЕ ===
            M = cv2.getRotationMatrix2D((cx, cy), base_angle, 1.0)

            rotated_poly = cv2.transform(poly_np, M)

            min_x = float(rotated_poly[:, 0, 0].min())
            max_x = float(rotated_poly[:, 0, 0].max())
            min_y = float(rotated_poly[:, 0, 1].min())
            max_y = float(rotated_poly[:, 0, 1].max())

            rect_w = max_x - min_x
            rect_h = max_y - min_y

            if rect_w < 1 or rect_h < 1:
                return self._crop_fallback(image, bbox)

            # === 4. PADDING (СТАБИЛЬНЫЙ) ===
            pad_ratio = 0.02
            pad_w = rect_w * pad_ratio
            pad_h = rect_h * pad_ratio

            tight_w = int(math.ceil(rect_w + pad_w * 2))
            tight_h = int(math.ceil(rect_h + pad_h * 2))

            if tight_w <= 1 or tight_h <= 1:
                return self._crop_fallback(image, bbox)

            # === 5. ЦЕНТРИРОВАНИЕ ===
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            M[0, 2] += (tight_w / 2.0) - center_x
            M[1, 2] += (tight_h / 2.0) - center_y

            # === 6. КРОП ===
            tight_crop = cv2.warpAffine(
                image,
                M,
                (tight_w, tight_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=NEUTRAL_BG,
            )

            if tight_crop.size == 0:
                return self._crop_fallback(image, bbox)

            tight_poly = cv2.transform(poly_np, M).reshape(-1, 2)

            # === 7. КАНВАС ПОД TARGET ASPECT ===
            target_h, target_w = self.input_size
            target_ar = target_w / target_h
            current_ar = tight_w / tight_h

            if current_ar > target_ar:
                final_w = tight_w
                final_h = int(round(tight_w / target_ar))
            else:
                final_h = tight_h
                final_w = int(round(tight_h * target_ar))

            final_w = max(2, final_w)
            final_h = max(2, final_h)

            canvas = np.full((final_h, final_w, 3), NEUTRAL_BG, dtype=np.uint8)

            # === 8. ЦЕНТР + СЛАБЫЙ SHIFT ===
            x_offset = (final_w - tight_w) // 2
            y_offset = (final_h - tight_h) // 2

            canvas[
                y_offset:y_offset + tight_h,
                x_offset:x_offset + tight_w
            ] = tight_crop

            final_poly = tight_poly + np.array([x_offset, y_offset])

            # === DEBUG ===
            if debug_mode:
                pts = final_poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
                
            return canvas, final_poly.tolist()

        except Exception as e:
            logger.warning(f"_crop_horizontal failed ({type(e).__name__}: {e}); falling back to bbox crop.")
            return self._crop_fallback(image, bbox)

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def _get_arcface_centroid_scores(
        self,
        emb_norm: torch.Tensor,
        top_k: int,
        valid_classes_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if self._arcface_centroids is None:
            raise RuntimeError(
                "ArcFace centroids are not available in this model. "
                "Use method='natural_centroid' or export model with arcface head."
            )
        
        B = emb_norm.shape[0]
        logger.debug(f"arcface_centroid scoring: batch_size={B}, embeddings shape={emb_norm.shape}")

        k_centers = self.config.k_centers
        total_centroids = self._arcface_centroids.shape[0]
        if total_centroids % k_centers != 0:
            raise ValueError(
                f"ArcFace centroid count ({total_centroids}) is not divisible by "
                f"k_centers ({k_centers}). Check config.k_centers or the model head."
            )
        num_classes = total_centroids // k_centers

        # IMPORTANT CONTRACT: ArcFace head weights must be stored in the order
        # [cls0_sub0, cls0_sub1, ..., cls0_subK, cls1_sub0, ..., clsN_subK].
        # If the training order differs, grouping below will silently produce wrong results.
        assert num_classes * k_centers == total_centroids, (
            f"Centroid layout sanity check failed: {num_classes} * {k_centers} != {total_centroids}"
        )

        similarities = torch.mm(emb_norm, self._arcface_centroids.t())
        logger.debug(f"Similarities matrix: {similarities.shape}, range=[{similarities.min():.4f}, {similarities.max():.4f}]")

        grouped_similarities = similarities.view(B, num_classes, k_centers)
        class_similarities, _ = torch.max(grouped_similarities, dim=2)

        if valid_classes_mask is not None:
            logger.debug(f"Applying class mask: {valid_classes_mask.sum().item()}/{len(valid_classes_mask)} valid classes.")
            class_similarities = class_similarities.masked_fill(
                ~valid_classes_mask.to(self.device).unsqueeze(0), float('-inf')
            )

        values, predicted_classes = torch.topk(class_similarities, k=top_k, dim=1, largest=True)
        return values.cpu().numpy(), predicted_classes.cpu().numpy()

    def _get_natural_centroid_scores(
        self,
        emb_norm: torch.Tensor,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        B = emb_norm.shape[0]
        logger.debug(f"natural_centroid scoring: batch_size={B}, embeddings shape={emb_norm.shape}")

        similarities = torch.mm(emb_norm, self._natural_centroids.t())  # (B, Num_Centroids)

        num_classes_from_mapping = (max(self.class_mapping.keys()) + 1) if self.class_mapping else 0
        num_classes_from_labels  = int(self._natural_labels.max().item()) + 1
        num_classes = max(num_classes_from_mapping, num_classes_from_labels)
        logger.debug(f"Aggregating {self._natural_centroids.shape[0]} centroids into {num_classes} classes.")

        class_similarities = torch.full((B, num_classes), float('-inf'), device=self.device)
        labels_expanded = self._natural_labels.unsqueeze(0).expand(B, -1)

        try:
            class_similarities.scatter_reduce_(1, labels_expanded, similarities, reduce="amax", include_self=False)
        except AttributeError:
            logger.warning("torch.scatter_reduce_ unavailable; falling back to loop-based aggregation (slower).")
            for i in range(num_classes):
                mask = (self._natural_labels == i)
                if mask.any():
                    class_similarities[:, i] = similarities[:, mask].max(dim=1)[0]

        class_similarities[class_similarities == float('-inf')] = -1.0

        values, predicted_classes = torch.topk(class_similarities, k=top_k, dim=1, largest=True)
        return values.cpu().numpy(), predicted_classes.cpu().numpy()

    # ------------------------------------------------------------------
    # Preprocessing & output builder
    # ------------------------------------------------------------------

    def _normalize_inputs(
        self,
        images: ImageLike | Sequence[ImageLike],
        bboxes: Optional[Sequence[Optional[List[float]]]],
        polys: Optional[Sequence[Optional[List[List[float]]]]],
    ) -> Tuple[List[ImageLike], List[Optional[List[float]]], List[Optional[List[List[float]]]]]:
        """Normalize single image / sequence inputs into aligned lists."""
        single = not isinstance(images, (list, tuple))
        imgs = [images] if single else list(images)
        bbs = [bboxes] if (single and bboxes is not None) else (bboxes if bboxes is not None else [None] * len(imgs))
        pls = [polys] if (single and polys is not None) else (polys if polys is not None else [None] * len(imgs))
        return imgs, bbs, pls

    def _preprocess(
        self,
        images: Sequence[ImageLike],
        bboxes: Optional[Sequence[Optional[List[float]]]] = None,
        polys: Optional[Sequence[Optional[List[List[float]]]]] = None,
    ) -> torch.Tensor:
        tensors = []
        batch_size = len(images)
        logger.debug(f"Preprocessing batch of {batch_size} image(s).")

        for i, img in enumerate(images):
            try:
                np_img = _to_numpy_rgb(img)
            except Exception as e:
                logger.error(f"Failed to convert image {i} to NumPy RGB: {e}")
                raise

            if bboxes is not None and polys is not None:
                np_img, _ = self._crop_horizontal(np_img, bboxes[i], polys[i])

            try:
                tensor_img = self.transform(image=np_img)["image"]
                tensors.append(tensor_img)
            except Exception as e:
                logger.error(f"Albumentations transform failed on image {i}: {e}")
                raise

        final_tensor = torch.stack(tensors).to(self.device)
        return final_tensor

    def _build_predictions(self, scores: np.ndarray, classes: np.ndarray) -> List[PredictionResult]:
        results: List[PredictionResult] = []
        for score, lbl in zip(scores, classes):
            lbl = int(lbl)
            info = self.class_mapping.get(lbl, {})
            name = info.get("label", f"Class {lbl}")
            results.append(PredictionResult(
                name=name,
                species_id=info.get("species_id"),
                distance=round(max(0.0, 1.0 - float(score)), 4),
                accuracy=float(score),
            ))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(
        self,
        images: ImageLike | Sequence[ImageLike],
        bboxes: Optional[Sequence[Optional[List[float]]]] = None,
        polys: Optional[Sequence[Optional[List[List[float]]]]] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        imgs, bbs, pls = self._normalize_inputs(images, bboxes, polys)
        single = not isinstance(images, (list, tuple))
        logger.info(f"Extracting embeddings for {len(imgs)} image(s)...")

        all_embs = []
        for start in range(0, len(imgs), batch_size):
            end = min(start + batch_size, len(imgs))
            batch = self._preprocess(imgs[start:end], bbs[start:end], pls[start:end])
            outputs = self.model(batch)
            emb = F.normalize(outputs[0], p=2, dim=1)

            all_embs.append(emb.cpu().numpy())

        result = np.concatenate(all_embs, axis=0)
        return result[0] if single else result

    @torch.no_grad()
    def predict(
        self,
        images: ImageLike | Sequence[ImageLike],
        bboxes: Optional[Sequence[Optional[List[float]]]] = None,
        polys: Optional[Sequence[Optional[List[List[float]]]]] = None,
        method: MethodType = "natural_centroid",
        valid_classes_mask: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> FishResult | List[FishResult]:
        logger.info(f"predict() method='{method}', debug={debug}")

        if method not in ("arcface_logits", "arcface_centroid", "natural_centroid"):
            raise ValueError(f"Unknown method '{method}'. Use 'arcface_logits', 'arcface_centroid', or 'natural_centroid'.")

        imgs, bbs, pls = self._normalize_inputs(images, bboxes, polys)
        single = not isinstance(images, (list, tuple))

        batch = self._preprocess(imgs, bbs, pls)

        debug_images = None
        if debug:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            unnorm = ((batch * std) + mean).clamp(0, 1) * 255.0
            debug_images = unnorm.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

        outputs = self.model(batch)

        emb_t = F.normalize(outputs[0], p=2, dim=1)
        logits_t = outputs[1]

        embeddings_np = emb_t.cpu().numpy()

        cfg = self.config
        if method == "arcface_logits":
            probs = F.softmax(logits_t, dim=-1)
            scores_t, classes_t = torch.topk(probs, k=cfg.max_unique_classes, dim=1)
            scores, classes = scores_t.cpu().numpy(), classes_t.cpu().numpy()
        elif method == "arcface_centroid":
            scores, classes = self._get_arcface_centroid_scores(emb_t, cfg.max_unique_classes, valid_classes_mask)
        elif method == "natural_centroid":
            scores, classes = self._get_natural_centroid_scores(emb_t, cfg.max_unique_classes)

        results = []
        for i in range(len(embeddings_np)):
            top_k = self._build_predictions(scores=scores[i], classes=classes[i])
            results.append(FishResult(
                top_k=top_k,
                embedding=embeddings_np[i] if cfg.return_emb else None,
                debug_image=debug_images[i] if debug else None,
            ))

        return results[0] if single else results

    @torch.no_grad()
    def predict_all_methods(
        self,
        images: ImageLike | Sequence[ImageLike],
        bboxes: Optional[Sequence[Optional[List[float]]]] = None,
        polys: Optional[Sequence[Optional[List[List[float]]]]] = None,
        valid_classes_mask: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> AllMethodsFishResult | List[AllMethodsFishResult]:
        """
        Same preprocessing and one ``model`` forward as ``predict``, then returns
        top-k for ``arcface_logits``, ``arcface_centroid`` (if ArcFace centroids
        exist; else ``None``), and ``natural_centroid`` per image.
        """
        logger.info("predict_all_methods(): single forward, all scoring branches")

        imgs, bbs, pls = self._normalize_inputs(images, bboxes, polys)
        single = not isinstance(images, (list, tuple))

        batch = self._preprocess(imgs, bbs, pls)

        debug_images = None
        if debug:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            unnorm = ((batch * std) + mean).clamp(0, 1) * 255.0
            debug_images = unnorm.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

        outputs = self.model(batch)
        emb_t = F.normalize(outputs[0], p=2, dim=1)
        logits_t = outputs[1]

        cfg = self.config
        k = cfg.max_unique_classes

        probs = F.softmax(logits_t, dim=-1)
        logits_scores_t, logits_classes_t = torch.topk(probs, k=k, dim=1)
        logits_scores = logits_scores_t.cpu().numpy()
        logits_classes = logits_classes_t.cpu().numpy()

        if self._arcface_centroids is not None:
            af_scores, af_classes = self._get_arcface_centroid_scores(
                emb_t, k, valid_classes_mask
            )
        else:
            af_scores = af_classes = None
            logger.debug("predict_all_methods: no ArcFace centroids; arcface_centroid omitted")

        nat_scores, nat_classes = self._get_natural_centroid_scores(emb_t, k)

        embeddings_np = emb_t.cpu().numpy()
        n = len(embeddings_np)

        def _fish(
            scores_row: np.ndarray,
            classes_row: np.ndarray,
            i: int,
        ) -> FishResult:
            top_k = self._build_predictions(scores=scores_row, classes=classes_row)
            return FishResult(
                top_k=top_k,
                embedding=embeddings_np[i] if cfg.return_emb else None,
                debug_image=debug_images[i] if debug else None,
            )

        out: List[AllMethodsFishResult] = []
        for i in range(n):
            af_res = (
                _fish(af_scores[i], af_classes[i], i)
                if af_scores is not None and af_classes is not None
                else None
            )
            out.append(
                AllMethodsFishResult(
                    arcface_logits=_fish(logits_scores[i], logits_classes[i], i),
                    arcface_centroid=af_res,
                    natural_centroid=_fish(nat_scores[i], nat_classes[i], i),
                )
            )

        return out[0] if single else out

    def predict_single(
        self,
        image: ImageLike,
        bbox: Optional[List[float]] = None,
        poly: Optional[List[List[float]]] = None,
        method: MethodType = "natural_centroid",
        valid_classes_mask: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> FishResult:
        result = self.predict(
            images=image,
            bboxes=bbox,
            polys=poly,
            method=method,
            valid_classes_mask=valid_classes_mask,
            debug=debug,
        )
        return result if isinstance(result, FishResult) else result[0]
        
    def predict_high_accuracy(
        self,
        image: ImageLike,
        bbox: Optional[List[float]] = None,
        poly: Optional[List[List[float]]] = None,
        method: MethodType = "natural_centroid",
        valid_classes_mask: Optional[torch.Tensor] = None,
        n_iterations: int = 5,
        aggregation: Literal["soft_voting", "embedding_mean"] = "soft_voting",
        original_weight: float = 1.0,
    ) -> FishResult:
        """
        High-accuracy prediction using Test-Time Augmentation (TTA) by varying the 
        sacrifice_factor (crop scale) and applying safe augmentations.
        """
        logger.info(
            f"Executing predict_high_accuracy: method='{method}', "
            f"iters={n_iterations}, agg={aggregation}"
        )

        np_img = _to_numpy_rgb(image)
        tensors = []

        # 1. Original image (no augmentation, base crop)
        if bbox is not None and poly is not None:
            cropped_orig, _ = self._crop_horizontal(np_img, bbox, poly)
        else:
            cropped_orig = np_img

        tensors.append(self.transform(image=cropped_orig)["image"])

        # 2. Augmented TTA views
        for _ in range(n_iterations - 1):
            tensors.append(self.tta_transform(image=cropped_orig)["image"])

        batch = torch.stack(tensors).to(self.device)
        logger.debug(f"TTA Forward Pass: Batch shape = {batch.shape} ({len(tensors)} views)")

        # 3. Inference for the entire batch
        with torch.no_grad():
            outputs = self.model(batch)
            emb_t = F.normalize(outputs[0], p=2, dim=1)
            logits_t = outputs[1]

            # -----------------------------------------------------------------
            # OPTION A: Embedding Averaging
            # -----------------------------------------------------------------
            if aggregation == "embedding_mean":
                logger.debug("Aggregating TTA results via 'embedding_mean'")
                mean_emb = emb_t.mean(dim=0, keepdim=True)
                emb_t_norm = F.normalize(mean_emb, p=2, dim=1)
                
                if method == "arcface_centroid":
                    scores, classes = self._get_arcface_centroid_scores(emb_t_norm, self.config.max_unique_classes, valid_classes_mask)
                elif method == "natural_centroid":
                    scores, classes = self._get_natural_centroid_scores(emb_t_norm, self.config.max_unique_classes)
                else: # arcface 
                    mean_logits = logits_t.mean(dim=0, keepdim=True)
                    probs = F.softmax(mean_logits, dim=-1)
                    scores_t, classes_t = torch.topk(probs, k=self.config.max_unique_classes, dim=1)
                    scores, classes = scores_t.cpu().numpy(), classes_t.cpu().numpy()
                    
                top_k = self._build_predictions(scores=scores[0], classes=classes[0])
                return FishResult(top_k=top_k, embedding=emb_t_norm[0].cpu().numpy() if self.config.return_emb else None)

            # -----------------------------------------------------------------
            # OPTION B: Soft Voting and Reranking
            # -----------------------------------------------------------------
            elif aggregation == "soft_voting":
                logger.debug("Aggregating TTA results via 'soft_voting'")
                if method == "arcface_centroid":
                    scores, classes = self._get_arcface_centroid_scores(emb_t, self.config.max_unique_classes, valid_classes_mask)
                elif method == "natural_centroid":
                    scores, classes = self._get_natural_centroid_scores(emb_t, self.config.max_unique_classes)
                else: # arcface
                    probs = F.softmax(logits_t, dim=-1)
                    scores_t, classes_t = torch.topk(probs, k=self.config.max_unique_classes, dim=1)
                    scores, classes = scores_t.cpu().numpy(), classes_t.cpu().numpy()

                class_scores = defaultdict(float)
                
                for i in range(len(scores)):
                    weight = original_weight if i == 0 else 1.0
                    for score, cls_id in zip(scores[i], classes[i]):
                        cls_id = int(cls_id)
                        class_scores[cls_id] += float(score) * weight
                        
                sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

                # One forward per batch row: row 0 uses original_weight, others 1.0.
                # (n_iterations is reserved for future multi-view TTA; it must not inflate the divisor.)
                total_weight = original_weight + max(0, len(scores) - 1)
                top_k_results = []
                for cls_id, total_score in sorted_classes[:self.config.max_unique_classes]:
                    avg_score = total_score / total_weight
                    info = self.class_mapping.get(cls_id, {})
                    name = info.get("label", f"Class {cls_id}")
                    
                    top_k_results.append(PredictionResult(
                        name=name,
                        species_id=info.get("species_id"),
                        distance=round(1.0 - avg_score, 4),
                        accuracy=avg_score,
                    ))
                    
                mean_emb = F.normalize(emb_t.mean(dim=0, keepdim=True), p=2, dim=1)
                return FishResult(
                    top_k=top_k_results,
                    embedding=mean_emb[0].cpu().numpy() if self.config.return_emb else None,
                )

            raise ValueError(
                f"Unknown aggregation '{aggregation}'; use 'soft_voting' or 'embedding_mean'."
            )

    def warmup(self, num_iterations: int = 3) -> None:
        """
        Warms up the model to allocate memory and optimize cuDNN benchmarks.
        
        Args:
            num_iterations (int): Number of dummy forward passes to execute.
        """
        logger.info(f"Warming up model for {num_iterations} iteration(s) on device '{self.device}'...")
        
        # Create a dummy tensor of the expected input size
        dummy = torch.randn(1, 3, self.input_size[0], self.input_size[1]).to(self.device)
        
        with torch.no_grad():
            for i in range(num_iterations):
                logger.debug(f"Warmup iteration {i + 1}/{num_iterations}")
                
                # For TorchScript/JIT models, simply call forward without kwargs
                self.model(dummy)
                
        # If using a GPU, it is good practice to wait for all computations to finish
        if "cuda" in self.device:
            torch.cuda.synchronize()
            
        logger.info("Warmup complete. Model is ready for inference.")