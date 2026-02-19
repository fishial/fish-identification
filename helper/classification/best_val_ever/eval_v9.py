import argparse
import json
import os
from pathlib import Path

import fiftyone as fo
from PIL import Image, ImageDraw
import numpy as np
from model_prod_v93.inference import EmbeddingClassifier
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
    Возвращает изображение, где объект внутри полигона виден,
    а остальное прозрачное (маска объекта).
    """
    if not polygon:
        return None

    # Создаем маску для полигона
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)

    # Создаем новое изображение с прозрачным фоном
    result = Image.new("RGB", image.size, (0, 0, 0, 0))
    result.paste(image, mask=mask)

    # Вырезаем прямоугольник вокруг полигона для компактного размера
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
    return cropped_image

classifier_config = {
        'model':{
            'path': 'model_prod_v93/model.ckpt',
            'device': 'cuda'
        },
        'dataset':{
            'path': 'model_prod_v93/database.pt'
        },
        'log_level': 'INFO'
    }

EmbeddingClassifier = EmbeddingClassifier(classifier_config)
dataset = fo.load_dataset("segmentation_dataset_v0.10_with_meta")

results = []

total_annotations = 0
total_first_correct = 0

with tqdm(dataset, desc="Evaluating", unit="sample") as progress:
    for sample in progress:
        image = Image.open(sample['filepath']).convert("RGB")
        width, height = image.size

        image_id = sample['image_id']

        for idx, polyline in enumerate(sample['General body shape']['polylines']):
            drawn_fish_id = polyline['drawn_fish_id']
            ann_id = polyline['ann_id']

            if not polyline.points:
                continue

            polygon = list(flatten_polyline_points(polyline.points, width, height))
            if not polygon:
                continue

            gt_label = polyline.label or "unnamed"

            object_pil_img = crop_polygon(image, polygon)
            if object_pil_img is None:
                continue

            result = EmbeddingClassifier.inference_numpy(np.array(object_pil_img))

            sorted_result = sorted(result, key=lambda item: item.accuracy if item.accuracy is not None else 0.0, reverse=True)
            names = [str(i.name) for i in sorted_result]
            accuracy = [round(float(i.accuracy or 0.0), 3) for i in sorted_result]

            total_annotations += 1
            is_top1_correct = len(sorted_result) > 0 and str(sorted_result[0].name) == gt_label
            if is_top1_correct:
                total_first_correct += 1

            if total_annotations > 0:
                progress.set_postfix({
                    "overall_acc": f"{(total_first_correct / total_annotations * 100):.2f}%",
                    "top1_rate": f"{total_first_correct}/{total_annotations}"
                })
            
            results.append({
                "image_id": image_id,
                "drawn_fish_id": drawn_fish_id,
                "ann_id": ann_id,
                "gt_label": gt_label,
                "names": names,
                "accuracy": accuracy,
            })

output_path = Path("inference_results_v93.json")
output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"Saved {len(results)} annotations to {output_path}")