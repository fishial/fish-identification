#!/usr/bin/env python3
"""
SAM3 Fish Segmentation - FiftyOne Integration

This script uses SAM3 (Segment Anything Model 3) to segment fish
via text prompts and saves the results as polygons in FiftyOne.

Features:
- ✅ Loading data from FiftyOne
- ✅ Automatic segmentation via text prompt (no box prompts required)
- ✅ Conversion of masks to polygons (vector format)
- ✅ Saving Polylines to FiftyOne (no masks or bboxes)
- ✅ Results visualization
- ✅ Progress bar with tqdm
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Sequence
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Progress bar
from tqdm import tqdm

# FiftyOne
import fiftyone as fo
from fiftyone import ViewField as F

# PIL
from PIL import Image, ExifTags

# SAM3
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("❌ SAM3 is not installed!")
    print("Install SAM3:")
    print("  pip install sam3  # or follow the instructions from the SAM3 repository")
    sys.exit(1)


# ============== CONFIGURATION ==============

# FiftyOne dataset
FIFTYONE_DATASET_NAME = "segmentation_dataset_v0.10_AI_GEN_ALL_VERIFIED"

# SAM3 model
# Available models: sam3_default_gpu, sam3_cpu_preview, sam3_interactive
SAM3_MODEL_KEY = "sam3_default_gpu"  # Use "sam3_cpu_preview" for CPU
SAM3_CHECKPOINT = None  # None for auto-download from HuggingFace, or path to checkpoint

# Text prompt for SAM3
TEXT_PROMPT = "a fish"  # Primary text prompt for fish detection

# Label for segmentation
SEGMENTATION_LABEL = "Fish"

# MODE: Automatic segmentation using text prompt only
# SAM3 will find all objects matching TEXT_PROMPT without box prompts

# Field to save segmentation results
PREDICTIONS_FIELD = f'sam3_segmentation'

# Processing parameters
CONFIDENCE_THRESHOLD = 0.6  # Minimum prompt confidence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Polygon parameters
POLYGON_SIMPLIFICATION = 0.0001  # Polygon simplification (0.0001 = max detail, 0.01 = min)
MIN_POLYGON_POINTS = 3  # Minimum number of points in a polygon
DISABLE_SIMPLIFICATION = True  # True = no simplification (maximum points!)

# Mask orientation correction
TRANSPOSE_MASK = False  # Transpose mask (swap H and W)
FLIP_VERTICAL = False   # Flip vertically
FLIP_HORIZONTAL = False # Flip horizontally

# Visualization
VISUALIZE_SAMPLES = 5  # Number of samples to visualize
SAVE_VISUALIZATIONS = False
VISUALIZE_POLYGONS = False  # Show polygons on visualization
OUTPUT_DIR = Path("./sam3_outputs")

# ============================================


@dataclass
class BoxPrompt:
    """Box prompt for SAM3"""
    name: str
    xyxy: Tuple[float, float, float, float]
    is_positive: bool = True


def xyxy_to_cxcywh_norm(box_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    """Converts bbox from xyxy to normalized cxcywh format for SAM3"""
    x0, y0, x1, y1 = box_xyxy
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    cx = x0 + w / 2
    cy = y0 + h / 2
    return [cx / width, cy / height, w / width, h / height]


def get_image_size_with_exif(image_path: str) -> Tuple[int, int]:
    """
    Gets image size considering EXIF orientation
    
    Returns:
        (width, height) after applying EXIF orientation
    """
    image = Image.open(image_path)
    image = correct_image_orientation(image)
    return image.size  # (width, height)


def correct_image_orientation(image: Image.Image) -> Image.Image:
    """
    Applies EXIF orientation to the image
    
    Many cameras (especially phones) save images in one orientation but 
    record the correct orientation in EXIF metadata. 
    This function reads EXIF and rotates the image accordingly.
    """
    try:
        exif = image.getexif()
        if exif is None:
            return image
        
        # Find the orientation tag
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == 'Orientation':
                orientation_key = key
                break
        
        if orientation_key is None or orientation_key not in exif:
            return image
        
        orientation = exif[orientation_key]
        
        # Apply transformations according to EXIF
        if orientation == 1:
            pass
        elif orientation == 2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.rotate(90, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 7:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
        
        return image
    except (AttributeError, KeyError, IndexError):
        # If EXIF is corrupted or missing, return image as is
        return image


def mask_to_polygons(mask: np.ndarray, min_points: int = 8) -> List[np.ndarray]:
    """Converts a binary mask to a list of polygons (one per connected component)"""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) < min_points:
            continue
        polys.append(cnt.reshape(-1, 2))
    return polys


class SAM3Segmenter:
    """Class for image segmentation using SAM3"""
    
    def __init__(
        self,
        model_key: str = "sam3_default_gpu",
        confidence: float = 0.5,
        checkpoint_path: Optional[str] = None,
        text_prompt: str = "a fish",
        transpose_mask: bool = True,
        flip_vertical: bool = False,
        flip_horizontal: bool = False
    ):
        """
        Initializes the SAM3 model
        
        Args:
            model_key: Model key (sam3_default_gpu, sam3_cpu_preview, sam3_interactive)
            confidence: Confidence threshold
            checkpoint_path: Path to checkpoint (None for auto-download)
            text_prompt: Text prompt for detection
        """
        self.model_key = model_key
        self.confidence = confidence
        self.checkpoint_path = checkpoint_path
        self.text_prompt = text_prompt
        self.transpose_mask = transpose_mask
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        
        print(f"🔧 Loading SAM3 model: {model_key}")
        if checkpoint_path:
            print(f"📦 Checkpoint: {checkpoint_path}")
        else:
            print(f"📦 Auto-loading from HuggingFace")
        print(f"🎯 Text prompt: '{text_prompt}'")
        print(f"📊 Confidence threshold: {confidence}")
        print(f"🔄 Orientation corrections:")
        print(f"   - Transpose: {transpose_mask}")
        print(f"   - Flip vertical: {flip_vertical}")
        print(f"   - Flip horizontal: {flip_horizontal}")
        print()
        
        # Determine model parameters
        device = "cuda" if model_key != "sam3_cpu_preview" and torch.cuda.is_available() else "cpu"
        
        model_kwargs = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "load_from_HF": checkpoint_path is None,
            "enable_segmentation": True,
            "enable_inst_interactivity": model_key == "sam3_interactive",
            "eval_mode": True,
        }
        
        print(f"🖥️  Device: {device}")
        
        # Load model
        self.model = build_sam3_image_model(**model_kwargs)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence)
        self.device = device
        
    def segment_from_text_prompt(
        self,
        image_path: str,
        debug: bool = False
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Segments image using text prompt only (no box prompts)
        
        Args:
            image_path: Path to image
            
        Returns:
            masks: List of segmentation masks (binary arrays)
            scores: List of scores for each mask
        """
        # Load image with automatic EXIF orientation correction
        image = Image.open(image_path).convert("RGB")
        image = correct_image_orientation(image)  # ← IMPORTANT: Apply EXIF!
        width, height = image.size
        
        if debug:
            print(f"\n  📷 Image after EXIF correction: {width}x{height}", file=sys.stderr)
        
        # Set image
        state = self.processor.set_image(image=image, state={})
        
        # Add ONLY text prompt
        if self.text_prompt:
            state = self.processor.set_text_prompt(prompt=self.text_prompt, state=state)
        else:
            state = self.processor.set_text_prompt(prompt="object", state=state)
        
        masks = []
        scores = []
        
        # SAM3 returns masks in different fields depending on the mode
        masks_tensor = None
        
        # Option 1: Directly in state["masks"]
        if "masks" in state and state["masks"] is not None:
            masks_tensor = state["masks"]
        # Option 2: In prediction results
        elif "pred_masks" in state and state["pred_masks"] is not None:
            masks_tensor = state["pred_masks"]
        # Option 3: In output
        elif "output" in state and state["output"] is not None:
            output = state["output"]
            if isinstance(output, dict):
                if "masks" in output:
                    masks_tensor = output["masks"]
                elif "pred_masks" in output:
                    masks_tensor = output["pred_masks"]
        
        if masks_tensor is not None:
            if isinstance(masks_tensor, torch.Tensor):
                masks_np = masks_tensor.cpu().numpy()
                
                # Handle different dimensions: (N, H, W), (N, 1, H, W), etc.
                if masks_np.ndim == 4:
                    if masks_np.shape[0] == 1:
                        masks_np = masks_np[0]
                    else:
                        masks_np = masks_np[:, 0]
                elif masks_np.ndim == 3:
                    pass
                elif masks_np.ndim == 2:
                    masks_np = masks_np[np.newaxis, ...]
                
                for idx, mask in enumerate(masks_np):
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    
                    if debug and idx == 0:
                        print(f"\n  🔍 DEBUG (pre-transform):", file=sys.stderr)
                        print(f"     Image: {width}x{height} (WxH)", file=sys.stderr)
                        print(f"     SAM3 Mask:  {mask_binary.shape[1]}x{mask_binary.shape[0]} (WxH)", file=sys.stderr)
                        print(f"     Total masks: {len(masks_np)}", file=sys.stderr)
                    
                    # ORIENTATION CORRECTION (Apply BEFORE resize!)
                    if self.transpose_mask:
                        mask_binary = np.transpose(mask_binary)
                        if debug and idx == 0:
                            print(f"     ✓ Transpose: {mask_binary.shape[1]}x{mask_binary.shape[0]} (WxH)", file=sys.stderr)
                    
                    if self.flip_vertical:
                        mask_binary = np.flipud(mask_binary)
                        if debug and idx == 0:
                            print(f"     ✓ Flip vertical", file=sys.stderr)
                    
                    if self.flip_horizontal:
                        mask_binary = np.fliplr(mask_binary)
                        if debug and idx == 0:
                            print(f"     ✓ Flip horizontal", file=sys.stderr)
                    
                    # Resize mask to match image size if necessary
                    if mask_binary.shape[0] != height or mask_binary.shape[1] != width:
                        mask_resized = cv2.resize(
                            mask_binary,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST
                        )
                        if debug and idx == 0:
                            print(f"     After resize:    {mask_resized.shape[1]}x{mask_resized.shape[0]} (WxH)", file=sys.stderr)
                        masks.append(mask_resized)
                    else:
                        masks.append(mask_binary)
            
            scores = [self.confidence] * len(masks)
        else:
            print(f"\n⚠️  WARNING: No masks found in state!", file=sys.stderr)
            print(f"   Available keys: {list(state.keys())}", file=sys.stderr)
        
        return masks, scores


def extract_prompts_from_sample(
    sample: fo.Sample,
    prompt_field: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts bounding boxes as prompts from a FiftyOne sample
    """
    img = cv2.imread(sample.filepath)
    h, w = img.shape[:2]
    
    if prompt_field and prompt_field in sample:
        detections = sample[prompt_field]
    elif "ground_truth" in sample:
        detections = sample["ground_truth"]
    elif "detections" in sample:
        detections = sample["detections"]
    else:
        return np.array([]), np.array([])
    
    if detections is None or len(detections.detections) == 0:
        return np.array([]), np.array([])
    
    boxes = []
    confidences = []
    
    for det in detections.detections:
        x, y, width, height = det.bounding_box
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + width) * w)
        y2 = int((y + height) * h)
        
        boxes.append([x1, y1, x2, y2])
        confidences.append(det.confidence if hasattr(det, 'confidence') and det.confidence else 1.0)
    
    return np.array(boxes), np.array(confidences)


def masks_to_fiftyone_polylines(
    masks: List[np.ndarray],
    scores: List[float],
    label: str,
    image_width: int,
    image_height: int,
    simplification: float = 0.005,
    min_points: int = 8,
    disable_simplification: bool = False
) -> fo.Polylines:
    """
    Converts SAM3 masks to FiftyOne Polylines (polygons only)
    """
    all_polylines = []
    
    for mask, score in zip(masks, scores):
        polygons = mask_to_polygons(mask, min_points=min_points)
        
        if len(polygons) == 0:
            continue
        
        # Get the largest polygon (main fish contour)
        largest_polygon = max(polygons, key=lambda p: cv2.contourArea(p.reshape(-1, 1, 2)))
        
        if disable_simplification:
            approx = largest_polygon.reshape(-1, 1, 2)
        else:
            epsilon = simplification * cv2.arcLength(largest_polygon.reshape(-1, 1, 2), True)
            approx = cv2.approxPolyDP(largest_polygon.reshape(-1, 1, 2), epsilon, True)
        
        # Convert to relative coordinates
        polygon_points = []
        for point in approx:
            px, py = point[0]
            polygon_points.append([
                float(px / image_width),
                float(py / image_height)
            ])
        
        if len(polygon_points) >= 3:
            polyline = fo.Polyline(
                label=label,
                points=[polygon_points],
                confidence=score,
                closed=False,
                filled=True
            )
            all_polylines.append(polyline)
    
    return fo.Polylines(polylines=all_polylines)


def visualize_segmentation(
    image_path: str,
    masks: List[np.ndarray],
    boxes: np.ndarray,
    output_path: Optional[str] = None,
    show_polygons: bool = True
):
    """
    Visualizes segmentation results
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3 if show_polygons else 2, figsize=(30 if show_polygons else 20, 10))
    
    # Original image
    axes[0].imshow(image_rgb)
    if len(boxes) > 0:
        axes[0].set_title("Prompts (Bounding Boxes)", fontsize=16)
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
    else:
        axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')
    
    # Masks
    axes[1].imshow(image_rgb)
    axes[1].set_title("SAM3 Segmentation (Masks)", fontsize=16)
    for i, mask in enumerate(masks):
        color = plt.cm.rainbow(i / len(masks))[:3]
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[mask > 0] = [*color, 0.5]
        axes[1].imshow(colored_mask)
    axes[1].axis('off')
    
    # Polygons
    if show_polygons:
        axes[2].imshow(image_rgb)
        axes[2].set_title("SAM3 Segmentation (Polygons)", fontsize=16)
        for i, mask in enumerate(masks):
            polygons = mask_to_polygons(mask, min_points=MIN_POLYGON_POINTS)
            color = plt.cm.rainbow(i / len(masks))[:3]
            for polygon in polygons:
                epsilon = POLYGON_SIMPLIFICATION * cv2.arcLength(polygon.reshape(-1, 1, 2), True)
                approx = cv2.approxPolyDP(polygon.reshape(-1, 1, 2), epsilon, True)
                if len(approx) >= 3:
                    poly_patch = Polygon(approx.reshape(-1, 2), closed=False, linewidth=2, edgecolor=color, facecolor=(*color, 0.3), fill=True)
                    axes[2].add_patch(poly_patch)
        axes[2].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Visualization saved: {output_path}")
    plt.close()


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("SAM3 Fish Segmentation - FiftyOne Integration")
    print("=" * 80)
    print()
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"📂 Loading FiftyOne dataset: {FIFTYONE_DATASET_NAME}")
    try:
        dataset = fo.load_dataset(FIFTYONE_DATASET_NAME)
    except ValueError:
        print(f"❌ Dataset '{FIFTYONE_DATASET_NAME}' not found!")
        sys.exit(1)
    
    test_view = dataset
    print(f"📊 Test images found: {len(test_view)}")
    
    try:
        segmenter = SAM3Segmenter(
            model_key=SAM3_MODEL_KEY,
            confidence=CONFIDENCE_THRESHOLD,
            checkpoint_path=SAM3_CHECKPOINT,
            text_prompt=TEXT_PROMPT,
            transpose_mask=TRANSPOSE_MASK,
            flip_vertical=FLIP_VERTICAL,
            flip_horizontal=FLIP_HORIZONTAL
        )
    except Exception as e:
        print(f"❌ Error loading SAM3: {e}")
        sys.exit(1)
    
    print(f"🎯 Mode: Automatic segmentation via text prompt")
    print(f"💾 Results will be saved to: '{PREDICTIONS_FIELD}'")
    print()
    
    total_masks = 0
    processed_samples = 0
    skipped_samples = 0
    
    with tqdm(
        total=len(test_view),
        desc="🎭 Segmentation",
        unit="img",
        ncols=100
    ) as pbar:
        
        for i, sample in enumerate(test_view, 1):
            try:
                masks, scores = segmenter.segment_from_text_prompt(sample.filepath, debug=(i <= 3))
                
                if len(masks) == 0:
                    skipped_samples += 1
                    pbar.update(1)
                    continue
                
                w, h = get_image_size_with_exif(sample.filepath)
                
                polylines = masks_to_fiftyone_polylines(
                    masks=masks, scores=scores, label=SEGMENTATION_LABEL,
                    image_width=w, image_height=h,
                    simplification=POLYGON_SIMPLIFICATION,
                    min_points=MIN_POLYGON_POINTS,
                    disable_simplification=DISABLE_SIMPLIFICATION
                )
                
                sample[PREDICTIONS_FIELD] = polylines
                sample.save()
                
                total_masks += len(masks)
                processed_samples += 1
                pbar.update(1)
                pbar.set_postfix({'masks': total_masks, 'skipped': skipped_samples})
                
                if SAVE_VISUALIZATIONS and i <= VISUALIZE_SAMPLES:
                    output_path = OUTPUT_DIR / f"sample_{i:04d}.png"
                    visualize_segmentation(sample.filepath, masks, np.array([]), str(output_path), show_polygons=VISUALIZE_POLYGONS)
                
            except Exception as e:
                pbar.write(f"⚠️  Error processing {sample.filepath}: {e}")
                skipped_samples += 1
                pbar.update(1)
    
    print()
    print("=" * 80)
    print("✅ SAM3 Segmentation Complete!")
    print("=" * 80)
    print(f"📊 Samples processed: {processed_samples}/{len(test_view)}")
    print(f"🎭 Total polygons created: {total_masks}")
    print(f"⏭️  Samples skipped: {skipped_samples}")
    print(f"💾 Results saved in field: '{PREDICTIONS_FIELD}'")


if __name__ == "__main__":
    main()