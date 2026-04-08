# -*- coding: utf-8 -*-
import logging
import random
import math
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = logging.getLogger(__name__)

# After ImageNet normalization, this will become ≈ (0, 0, 0) — an ideal neutral background.
NEUTRAL_BG = (124, 116, 104)


class FishialDatasetOnlineCutting(Dataset):
    """Dataset for online fish cropping with MBR-based horizontal alignment."""

    def __init__(
        self,
        records: list,
        img_size: tuple, # Expected (H, W), e.g., (154, 518)
        train_state: bool = False,
        transform=None,
        instance_data: bool = False,
        bbox_padding_limit: float = 0.15,
        bg_removal_prob: float = 0.0,
        alignment_method: str = 'ellipse',  # 'minar' | 'ellipse' | 'fitline'
        angle_jitter_deg: float = 7.0,
        morphology_max_area_regression: float = 0.10,
        morphology_max_iter: int = 20,
    ):
        self.records = records
        self.img_size = img_size
        self.train_state = train_state
        self.instance_data = instance_data
        self.bbox_padding_limit = bbox_padding_limit
        self.bg_removal_prob = bg_removal_prob
        self.alignment_method = alignment_method
        self.angle_jitter_deg = angle_jitter_deg
        self.morphology_max_area_regression = morphology_max_area_regression
        self.morphology_max_iter = morphology_max_iter
        
        # Safe ID extraction (prevents crash if the key is missing)
        self._targets = [item.get('id_internal', -1) for item in self.records]

        self.transform = transform or A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        logger.info(f"Initialized FishialDatasetOnlineCutting with {len(self.records)} records.")
        logger.info(f"Settings: img_size={self.img_size}, train_state={self.train_state}, "
                    f"alignment_method='{self.alignment_method}', bg_removal_prob={self.bg_removal_prob}")

    @property
    def targets(self) -> list:
        return self._targets

    def get_by_drawn_fish_id(self, drawn_fish_id: str):
        logger.debug(f"Searching for record with drawn_fish_id='{drawn_fish_id}'")
        for idx, record in enumerate(self.records):
            if record.get('drawn_fish_id') == drawn_fish_id:
                return self.__getitem__(idx)
        
        logger.error(f"Record with drawn_fish_id='{drawn_fish_id}' not found.")
        raise ValueError(f"Record with drawn_fish_id='{drawn_fish_id}' not found in the dataset.")

    # ------------------------------------------------------------------
    # Cropping & Morphology
    # ------------------------------------------------------------------

    def _crop_fallback(
        self,
        image: np.ndarray,
        bbox: Optional[list],
    ) -> tuple[np.ndarray, list]:
        if not bbox or len(bbox) != 4:
            logger.warning("[Crop Fallback] Invalid or missing bbox. Returning original image.")
            return image, []
        
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2].copy()
        
        if crop.size == 0:
            logger.warning(f"[Crop Fallback] Empty crop produced for bbox {bbox}. Returning original image.")
            return image, []
            
        logger.debug(f"[Crop Fallback] Successful fallback crop. Shape: {crop.shape}")
        return crop, []

    def _get_orientation_angle(
        self,
        poly_np: np.ndarray,
    ) -> tuple[float, float, float]:
        pts   = poly_np.reshape(-1, 1, 2)
        pts2d = poly_np.reshape(-1, 2)

        cx, cy, angle = self._raw_angle(pts, pts2d, self.alignment_method)
        logger.debug(f"[Orientation] Raw angle calculated: {angle:.2f} at center ({cx:.2f}, {cy:.2f})")

        angle = angle % 180.0
        if angle > 90.0:
            angle -= 180.0

        M_test = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.transform(pts, M_test).reshape(-1, 2)
        rw = float(rotated[:, 0].max() - rotated[:, 0].min())
        rh = float(rotated[:, 1].max() - rotated[:, 1].min())
        
        if rh > rw:
            angle += 90.0
            logger.debug(f"[Orientation] Height > Width. Adjusted angle by +90. New angle: {angle:.2f}")

        return float(cx), float(cy), float(angle)


    def _remove_thin_appendages_dynamic(
        self, 
        poly_np: np.ndarray, 
        max_area_regression: float = 0.10,
        debug_mode: bool = False
    ) -> np.ndarray:
        """
        Adaptively removes barbels and fin rays.
        """
        pts = poly_np.reshape(-1, 2)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        w, h = int(x_max - x_min), int(y_max - y_min)
        
        if w < 10 or h < 10:
            if debug_mode:
                logger.info("[Morphology] Skipped: Polygon too small.")
            return poly_np
            
        local_pts = np.round(pts - [x_min, y_min] + [10, 10]).astype(np.int32)
        
        mask = np.zeros((h + 20, w + 20), dtype=np.uint8)
        cv2.fillPoly(mask, [local_pts], 255)
        
        orig_area = cv2.countNonZero(mask)
        if orig_area == 0:
            return poly_np
            
        if debug_mode:
            logger.info(f"[Morphology] Start. Original Area: {orig_area} px. Target Max Degradation: {max_area_regression*100:.1f}%")

        best_mask = mask.copy()
        kernel_size = 3
        iteration = 0
        MAX_ITER = self.morphology_max_iter

        while True:
            iteration += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            new_area = cv2.countNonZero(opened)

            degradation = (orig_area - new_area) / float(orig_area)

            if debug_mode:
                logger.info(f"  -> Iter {iteration} | Kernel {kernel_size}x{kernel_size} | "
                            f"Area: {new_area} px | Degraded: {degradation*100:.2f}%")

            if degradation > max_area_regression or new_area == 0:
                if debug_mode:
                    logger.info(f"[Morphology] STOP. Degradation {degradation*100:.2f}% > {max_area_regression*100:.1f}%. "
                                f"Reverting to previous mask (Kernel {max(1, kernel_size-2)}).")
                break

            best_mask = opened.copy()
            kernel_size += 2

            if kernel_size > min(w, h) / 2 or iteration >= MAX_ITER:
                if debug_mode:
                    reason = "Max kernel size reached" if kernel_size > min(w, h) / 2 else f"Max iterations ({MAX_ITER}) reached"
                    logger.info(f"[Morphology] STOP. {reason}.")
                break

        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return poly_np
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        # === FIX 1: JUNK CONTOUR GUARD ===
        # If after morphology only a point or line remains (less than 3 points),
        # OpenCV will not be able to construct a rectangle/ellipse. Return the original.
        if len(largest_contour) < 3:
            if debug_mode:
                logger.warning("[Morphology] Largest contour has < 3 points. Reverting to original polygon.")
            return poly_np

        core_poly = largest_contour.astype(np.float32).reshape(-1, 1, 2)
        core_poly = core_poly - [10, 10] + [x_min, y_min]
        
        return core_poly


    def _raw_angle(
        self,
        pts: np.ndarray,    # (-1, 1, 2)
        pts2d: np.ndarray,  # (-1, 2)
        method: str,
    ) -> tuple[float, float, float]:
        """Raw angle calculation without orientation check — each method has its own logic."""

        # === FIX 2: STRICT TYPE CASTING FOR OPENCV ===
        # OpenCV crashes if the type is not CV_32F. Ensure it's float32.
        pts = np.array(pts, dtype=np.float32)

        # ── ELLIPSE ──────────────────────────────────────────────────────────
        if method == 'ellipse' and len(pts2d) >= 5:
            try:
                (cx, cy), (MA, ma), angle = cv2.fitEllipse(pts)
                return float(cx), float(cy), float(angle - 90.0)
            except cv2.error as e:
                logger.debug(f"fitEllipse failed: {e}")

        # ── FITLINE ──────────────────────────────────────────────────────────
        if method == 'fitline':
            try:
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                cx = float(pts2d[:, 0].mean())
                cy = float(pts2d[:, 1].mean())
                
                # === FIX 3: EXTRACT VALUES FROM ARRAY ===
                # vy and vx are returned as numpy arrays of size (1,).
                # math.atan2 expects regular floats.
                angle = math.degrees(math.atan2(float(vy[0]), float(vx[0])))
                return cx, cy, angle
            except cv2.error as e:
                logger.debug(f"fitLine failed: {e}")

        # ── MINAR (default / fallback) ────────────────────────────────────────
        try:
            rect = cv2.minAreaRect(pts)
            (cx, cy), _, _ = rect
            box = cv2.boxPoints(rect)
            d01 = np.linalg.norm(box[0] - box[1])
            d12 = np.linalg.norm(box[1] - box[2])
            dx, dy = (box[1] - box[0]) if d01 > d12 else (box[2] - box[1])
            if dx < 0:
                dx, dy = -dx, -dy
            return float(cx), float(cy), float(math.degrees(math.atan2(dy, dx)))
        except cv2.error as e:
            # Absolute fallback if even minAreaRect fails (should not happen after protection)
            logger.error(f"minAreaRect failed critically: {e}")
            cx = float(pts2d[:, 0].mean())
            cy = float(pts2d[:, 1].mean())
            return cx, cy, 0.0


    def _crop_horizontal(
        self,
        image: np.ndarray,
        bbox: Optional[list],
        poly: Optional[list],
        debug_mode: bool = False,
    ) -> tuple[np.ndarray, list]:

        if not poly or len(poly) < 3:
            logger.warning("[Crop Horizontal] Invalid polygon (len < 3). Triggering fallback.")
            return self._crop_fallback(image, bbox)

        try:
            raw_poly_np = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)

            poly_np = self._remove_thin_appendages_dynamic(
                raw_poly_np,
                max_area_regression=self.morphology_max_area_regression,
                debug_mode=debug_mode,
            )

            cx, cy, base_angle = self._get_orientation_angle(poly_np)

            if self.train_state:
                angle_jitter = random.uniform(-self.angle_jitter_deg, self.angle_jitter_deg)
                base_angle += angle_jitter
                logger.debug(f"[Crop Horizontal] Training state active. Added angle jitter: {angle_jitter:.2f}")

            M = cv2.getRotationMatrix2D((cx, cy), base_angle, 1.0)
            rotated_poly = cv2.transform(poly_np, M)

            min_x = float(rotated_poly[:, 0, 0].min())
            max_x = float(rotated_poly[:, 0, 0].max())
            min_y = float(rotated_poly[:, 0, 1].min())
            max_y = float(rotated_poly[:, 0, 1].max())

            rect_w = max_x - min_x
            rect_h = max_y - min_y

            if rect_w < 1 or rect_h < 1:
                logger.warning("[Crop Horizontal] Rotated rect is too small (<1px). Triggering fallback.")
                return self._crop_fallback(image, bbox)

            pad_ratio = max(0.0, self.bbox_padding_limit) 
            pad_w = rect_w * pad_ratio
            pad_h = rect_h * pad_ratio

            tight_w = int(math.ceil(rect_w + pad_w * 2))
            tight_h = int(math.ceil(rect_h + pad_h * 2))

            if tight_w <= 1 or tight_h <= 1:
                logger.warning("[Crop Horizontal] Tight crop dims are <= 1px. Triggering fallback.")
                return self._crop_fallback(image, bbox)

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            M[0, 2] += (tight_w / 2.0) - center_x
            M[1, 2] += (tight_h / 2.0) - center_y

            tight_crop = cv2.warpAffine(
                image,
                M,
                (tight_w, tight_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=NEUTRAL_BG,
            )

            if tight_crop.size == 0:
                logger.warning("[Crop Horizontal] warpAffine produced empty array. Triggering fallback.")
                return self._crop_fallback(image, bbox)

            tight_poly = cv2.transform(poly_np, M).reshape(-1, 2)
            
            if debug_mode:
                raw_tight_poly = cv2.transform(raw_poly_np, M).reshape(-1, 2)

            target_h, target_w = self.img_size
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

            x_offset = (final_w - tight_w) // 2
            y_offset = (final_h - tight_h) // 2

            if self.train_state:
                shift_x = random.randint(-x_offset, x_offset) if x_offset > 0 else 0
                shift_y = random.randint(-y_offset, y_offset) if y_offset > 0 else 0
                x_offset += shift_x
                y_offset += shift_y
                logger.debug(f"[Crop Horizontal] Applied translation shift: x={shift_x}, y={shift_y}")

            canvas[
                y_offset:y_offset + tight_h,
                x_offset:x_offset + tight_w
            ] = tight_crop

            final_poly = tight_poly + np.array([x_offset, y_offset])

            if debug_mode:
                final_raw_poly = raw_tight_poly + np.array([x_offset, y_offset])
                pts_raw = final_raw_poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_raw], True, (255, 0, 0), 2)  
                
                pts_clean = final_poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_clean], True, (0, 255, 0), 2)  

            logger.debug(f"[Crop Horizontal] Success. Canvas shape: {canvas.shape}")
            return canvas, final_poly.tolist()

        except Exception as e:
            logger.error(f"[Crop Horizontal] Failed with exception: {e}", exc_info=True)
            return self._crop_fallback(image, bbox)

    # ------------------------------------------------------------------
    # Mask & helpers
    # ------------------------------------------------------------------
    def prepare_for_inference(self, image_rgb: np.ndarray, poly: list) -> torch.Tensor:
        logger.debug("[Inference] Preparing image for inference.")
        if poly and len(poly) >= 3:
            poly_np = np.array(poly)
            x1, y1 = np.min(poly_np, axis=0)
            x2, y2 = np.max(poly_np, axis=0)
            bbox = [float(x1), float(y1), float(x2), float(y2)]
        else:
            bbox = [0.0, 0.0, float(image_rgb.shape[1]), float(image_rgb.shape[0])]
            logger.debug("[Inference] No poly provided, using full image bbox.")

        image_crop, _ = self._crop_horizontal(
            image=image_rgb,
            bbox=bbox,
            poly=poly,
            debug_mode=False
        )

        transformed = self.transform(image=image_crop)
        image_tensor = transformed['image']
        input_tensor = image_tensor.unsqueeze(0).float()
        
        logger.debug(f"[Inference] Final tensor shape: {input_tensor.shape}")
        return input_tensor

    @staticmethod
    def _create_mask(h: int, w: int, poly_coords: list) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(poly_coords) >= 3:
            pts = np.array(poly_coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
        return mask

    @staticmethod
    def _meta_value(value, default):
        return default if value is None else value

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        max_retries = 10
        for attempt in range(max_retries):
            item_data = self.records[idx]
            filepath = item_data.get('filepath', 'UNKNOWN_PATH')
            logger.debug(f"[Dataset] Loading item {idx} from {filepath}")
            full_image = cv2.imread(filepath)
            if full_image is not None:
                break
            logger.warning(f"[Dataset] Failed to read image at {filepath}. Retry {attempt + 1}/{max_retries}.")
            idx = random.randint(0, len(self.records) - 1)
        else:
            raise RuntimeError(f"[Dataset] Could not load a valid image after {max_retries} retries.")

        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

        image_crop, poly_crop = self._crop_horizontal(
            full_image, item_data.get('bbox_xyxy'), item_data.get('poly'), debug_mode=False
        )

        mask = self._create_mask(image_crop.shape[0], image_crop.shape[1], poly_crop)

        if self.train_state and self.bg_removal_prob > 0.0 and mask.max() > 0 and random.random() < self.bg_removal_prob:
            logger.debug(f"[Dataset] Applying background removal (prob: {self.bg_removal_prob}).")
            bg = np.full_like(image_crop, NEUTRAL_BG)
            image_crop = np.where(mask[..., None] == 1, image_crop, bg)

        transformed = self.transform(image=image_crop, mask=mask)
        image_tensor = transformed['image']
        mask_tensor = transformed['mask'].unsqueeze(0).float()
        
        label_tensor = torch.tensor(item_data.get('id_internal', -1), dtype=torch.long)

        if self.instance_data:
            instance_info = {
                'annotation_id': self._meta_value(item_data.get('annotation_id'), -1),
                'drawn_fish_id': self._meta_value(item_data.get('drawn_fish_id'), ''),
                'image_id': self._meta_value(item_data.get('image_id'), ''),
                'species_id': self._meta_value(item_data.get('species_id'), -1),
            }
            logger.debug(f"[Dataset] Successfully loaded instance data for item {idx}.")
            return image_tensor, label_tensor, mask_tensor, instance_info

        logger.debug(f"[Dataset] Successfully loaded item {idx}.")
        return image_tensor, label_tensor, mask_tensor