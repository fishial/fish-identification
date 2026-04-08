import sys
sys.path.append('/home/andrew/Andrew/Fishial2402/fish-identification')

import json
from pathlib import Path

import fiftyone as fo
from PIL import Image, ImageDraw
import numpy as np

from module.classification_package.fish_inference import FishInferenceEngine
from tqdm import tqdm

def normalized_to_pixels(points, width, height):
    """Convert normalized coordinates into pixel positions."""
    for x, y in points:
        yield int(x * width), int(y * height)


def flatten_polyline_points(points_list, width, height):
    """Flatten the shape list so it can be drawn as a single polygon."""
    for part in points_list:
        yield from normalized_to_pixels(part, width, height)


def crop_polygon_mask(image, polygon):
    """
    Returns an image where the object inside the polygon is visible,
    and the rest is transparent (object mask).
    """
    if not polygon:
        return None

    # Create a mask for the polygon
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)

    # Create a new image with a transparent background
    result = Image.new("RGB", image.size, (0, 0, 0, 0))
    result.paste(image, mask=mask)

    # Crop a rectangle around the polygon for a compact size
    xs, ys = zip(*polygon)
    bbox = (
        max(int(min(xs)), 0),
        max(int(min(ys)), 0),
        min(int(max(xs)) + 1, image.width),
        min(int(max(ys)) + 1, image.height),
    )

    cropped_result = result.crop(bbox)
    return cropped_result

def crop_polygon(image, polygon):
    """Crop the image tightly around the polygon and preserve the mask."""
    if not polygon:
        return

    xs, ys = zip(*polygon)
    bbox = (
        max(int(min(xs)), 0),
        max(int(min(ys)), 0),
        min(int(max(xs)) + 1, image.width),
        min(int(max(ys)) + 1, image.height),
    )

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return

    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)

    cropped_image = image.crop(bbox)
    return cropped_image, bbox


OUTPUT_PATHS = {
    "arcface_logits": Path("dinov2_arcface_logits_production_v18_866.json"),
    "arcface_centroid": Path("dinov2_arcface_centroid_production_v18_866.json"),
    "natural_centroid": Path("dinov2_natural_centroid_production_v18_866.json"),
}

BUNDLE_PATH = '/home/andrew/Andrew/Fishial2402/fish-identification/notebooks/model_bundle.pt'

# Initialize the inference engine
EmbeddingClassifier = FishInferenceEngine.from_bundle(BUNDLE_PATH)
EmbeddingClassifier.warmup()

# Load the FiftyOne dataset
dataset = fo.load_dataset("segmentation_dataset_v0.11_with_meta")

results_by_method = {key: [] for key in OUTPUT_PATHS}

total_annotations = 0
total_first_correct = {key: 0 for key in OUTPUT_PATHS}


def _sorted_names_accuracy(fish_result):
    """Helper to extract and sort names and accuracies from the inference result."""
    if fish_result is None:
        return [], []
    sorted_result = sorted(
        fish_result.top_k,
        key=lambda item: item.accuracy if item.accuracy is not None else 0.0,
        reverse=True,
    )
    names = [str(i.name) for i in sorted_result]
    accuracy = [round(float(i.accuracy or 0.0), 3) for i in sorted_result]
    return names, accuracy


with tqdm(dataset, desc="Evaluating", unit="sample") as progress:
    for sample in progress:
        image = Image.open(sample['filepath']).convert("RGB")
        width, height = image.size

        image_id = sample['image_id']

        # Iterate through fish annotations in the sample
        for idx, polyline in enumerate(sample['General body shape']['polylines']):
            drawn_fish_id = polyline['drawn_fish_id']
            ann_id = polyline['ann_id']

            if not polyline.points:
                continue

            polygon = list(flatten_polyline_points(polyline.points, width, height))
            if not polygon:
                continue

            gt_label = polyline.label or "unnamed"

            # Crop the object based on the polygon
            object_pil_img, bbox = crop_polygon(image, polygon)

            # Predict using all available methods in the engine
            all_res = EmbeddingClassifier.predict_all_methods(
                images=np.array(image),
                bboxes=bbox,
                polys=polygon,
            )
            
            method_to_fish = {
                "arcface_logits": all_res.arcface_logits,
                "arcface_centroid": all_res.arcface_centroid,
                "natural_centroid": all_res.natural_centroid,
            }

            total_annotations += 1

            for method_key, fish_res in method_to_fish.items():
                names, accuracy = _sorted_names_accuracy(fish_res)
                # Check if the top-1 prediction matches the ground truth
                is_top1_correct = len(names) > 0 and names[0] == gt_label
                if is_top1_correct:
                    total_first_correct[method_key] += 1

                results_by_method[method_key].append({
                    "image_id": image_id,
                    "drawn_fish_id": drawn_fish_id,
                    "ann_id": ann_id,
                    "gt_label": gt_label,
                    "names": names,
                    "accuracy": accuracy,
                    "state": "",
                })

            # Update progress bar statistics
            if total_annotations > 0:
                c = total_annotations
                progress.set_postfix({
                    "logits": f"{total_first_correct['arcface_logits'] / c * 100:.1f}%",
                    "af_ctr": f"{total_first_correct['arcface_centroid'] / c * 100:.1f}%",
                    "nat_ctr": f"{total_first_correct['natural_centroid'] / c * 100:.1f}%",
                })

# Save results to JSON files
for method_key, out_path in OUTPUT_PATHS.items():
    rows = results_by_method[method_key]
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    n = total_annotations
    acc_pct = (total_first_correct[method_key] / n * 100) if n else 0.0
    print(f"Saved {len(rows)} annotations to {out_path} (top-1: {acc_pct:.2f}%)")