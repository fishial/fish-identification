#!/usr/bin/env python3
"""
Script to create a classification dataset from COCO annotations.

This script processes annotations, extracts image crops, normalizes polygon segmentation,
and creates a FiftyOne dataset with polyline annotations. All parameters (paths, dataset name, etc.)
are provided via command-line arguments.
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
from collections import defaultdict

import fiftyone as fo

# Modify sys.path to include the root directory containing 'fish-identification'
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
    print("SETUP: sys.path updated")

from module.classification_package.src.utils import read_json, save_json
from module.segmentation_package.src.utils import get_mask


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a classification dataset from COCO annotations"
    )
    parser.add_argument('-dp', '--dst_path', type=str, required=True,
                        help="Destination folder for annotation file and cropped images")
    parser.add_argument('-i', '--path_to_full_images', type=str, required=True,
                        help="Folder containing full images")
    parser.add_argument('-a', '--path_to_src_coco_json', type=str, required=True,
                        help="Path to the source COCO JSON file")
    parser.add_argument('-dsn', '--voxel_dataset_name', type=str, required=True,
                        help="Name for the voxel dataset")
    parser.add_argument("--min_eval_img", type=int, default=10,
                        help="Minimum number of evaluation images per class")
    parser.add_argument("--max_percent_eva_img", type=float, default=0.2,
                        help="Maximum percentage of evaluation images per class")
    parser.add_argument("--max_cnt_img_per_class", type=int, default=350,
                        help="Maximum number of images per class")
    parser.add_argument("--min_cnt_img", type=int, default=50,
                        help="Minimum number of images per class")
    parser.add_argument("--min_crop_size", type=int, default=80,
                        help="Minimum width/height for image crops (default: 80)")
    return parser.parse_args()


def get_valid_categories(data: dict) -> dict:
    """
    Extract valid categories for segmentation based on criteria:
    - Category name is 'General body shape'
    - Supercategory is not 'unknown'

    Args:
        data (dict): COCO formatted data.

    Returns:
        dict: Mapping of valid category id to its supercategory.
    """
    valid_category = {}
    for cat in data.get('categories', []):
        if cat.get('name') == 'General body shape' and cat.get('supercategory') != 'unknown':
            valid_category[cat.get('id')] = cat.get('supercategory')
    return valid_category


def fix_polygon(poly: list, shape: list) -> list:
    """
    Adjust polygon points to ensure they lie within image boundaries.

    Args:
        poly (list): List of (x, y) tuples representing polygon points.
        shape (list): [width, height] of the image.

    Returns:
        list: Clamped polygon points.
    """
    width, height = shape
    return [(min(max(0, x), width), min(max(0, y), height)) for x, y in poly]


def polygon_area(x: list, y: list) -> float:
    """
    Calculate the area of a polygon given x and y coordinates using the shoelace formula.

    Args:
        x (list): List of x coordinates.
        y (list): List of y coordinates.

    Returns:
        float: Area of the polygon.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def process_annotations(args: argparse.Namespace) -> None:
    """
    Process annotations from COCO JSON.
    Optimized to read each image only once.
    """
    voxel_dataset_name = args.voxel_dataset_name
    dst_path = args.dst_path
    path_to_src_coco_json = args.path_to_src_coco_json
    path_full_images = args.path_to_full_images
    min_cnt_img = args.min_cnt_img
    min_crop_size = args.min_crop_size

    # List files in the images folder and read JSON data
    # Note: os.walk can be slow on huge directories, but we need to verify file existence.
    print(f"Scanning images directory: {path_full_images} ...")
    list_of_files = set(next(os.walk(path_full_images))[2])
    print(f"Total files in folder: {len(list_of_files)}")

    print(f"Reading JSON: {path_to_src_coco_json} ...")
    data_full = read_json(path_to_src_coco_json)

    # Create destination folder for cropped images
    folder_to_save_files = os.path.join(dst_path, 'images')
    os.makedirs(folder_to_save_files, exist_ok=True)

    # Filter valid images (exist on disk and not flagged as invalid)
    valid_images = {}
    
    for img in data_full.get('images', []):
        if img.get('is_invalid'):
            continue
        if img.get('file_name') in list_of_files:
            valid_images[img.get('id')] = img
        else:
            print('file does not exist: ', img.get('file_name'))
    print(f"Number of valid images in JSON: {len(valid_images)}")

    # Get valid categories for processing annotations
    valid_category = get_valid_categories(data_full)
    
    # Initialize dictionary to collect annotations per category (using supercategory as key)
    annotations_by_category = defaultdict(list)

    print("Filtering annotations...")
    
    # Statistics for skipped images per class
    # Structure: { class_name: { 'reason': count, ... } }
    skipped_stats = defaultdict(lambda: defaultdict(int))
    
    # Process each annotation logic - first pass to filter by counts
    for ann in data_full.get('annotations', []):
        cat_id = ann.get('category_id')
        # We only care about statistics for categories we MIGHT have accepted (General body shape + valid supercategory)
        # If it's not even in valid_category, it's likely irrelevant data or 'unknown' which we implicitly drop.
        # But to be safe, if we can map it, we track it.
        
        cat_name = valid_category.get(cat_id)
        if not cat_name:
            continue

        try:
            if ann.get('image_id') not in valid_images:
                skipped_stats[cat_name]['invalid_image_reference'] += 1
                continue
        except Exception:
            continue

        # Convert segmentation into list of (x, y) tuples
        seg = ann.get('segmentation', [])
        if seg:
            # Assuming format [x1, y1, x2, y2, ...] or [[x1, y1, ...]]
            if isinstance(seg[0], list):
                poly_list = seg[0]
            else:
                poly_list = seg
            
            points = [(int(poly_list[i]), int(poly_list[i + 1])) for i in range(0, len(poly_list), 2)]
            ann['segmentation'] = points
        else:
            skipped_stats[cat_name]['empty_segmentation'] += 1
            continue

        # Add extra info
        fishial_extra = valid_images[ann.get('image_id')].get('fishial_extra', {})
        ann['include_in_odm'] = fishial_extra.get('include_in_odm', False)
        
        annotations_by_category[cat_name].append(ann)

    # Remove categories with fewer than min_cnt_img annotations
    categories_to_remove = [cat for cat, anns in annotations_by_category.items() if len(anns) < min_cnt_img]
    for cat in categories_to_remove:
        count = len(annotations_by_category[cat])
        skipped_stats[cat]['below_min_count'] += count
        print(skipped_stats[cat])
        del annotations_by_category[cat]
    
    
    print(f"\nTotal approved categories: {len(annotations_by_category)}")

    # Create label mapping
    label_mapping = {label: idx for idx, label in enumerate(annotations_by_category.keys())}
    print(f"Label mapping size: {len(label_mapping)}")

    # Group annotations by IMAGE ID for efficient processing
    annotations_by_image = defaultdict(list)
    total_annotations = 0

    for cat_name, anns in annotations_by_category.items():
        # Sort by ODM flag if needed, though grouping by image changes order slightly
        # We will keep the original logic's attribute assignment
        for ann in anns:
            ann['id_internal'] = label_mapping.get(cat_name)
            ann['label'] = cat_name
            annotations_by_image[ann.get('image_id')].append(ann)
            total_annotations += 1
            
    print(f"Ready to process {total_annotations} annotations across {len(annotations_by_image)} images.")

    records = []
    samples = []
    
    processed_count = 0
    images_count = 0
    
    # Process images one by one
    for image_id, image_anns in annotations_by_image.items():
        images_count += 1
        print(f"Processing image {images_count}/{len(annotations_by_image)} (Anns: {processed_count})", end='\r')

        img_info = valid_images[image_id]
        img_path = os.path.join(path_full_images, img_info['file_name'])
        
        # Read image ONCE
        img = cv2.imread(img_path)
        if img is None:
            print(f"\nError: Could not read image {img_path}")
            # Log skipped annotations due to image load error
            for ann in image_anns:
                skipped_stats[ann['label']]['image_load_error'] += 1
            continue
            
        height, width = img.shape[:2]

        # Process all annotations for this image
        for ann_inst in image_anns:
            # Fix polygon
            ann_inst['segmentation'] = fix_polygon(ann_inst['segmentation'], [width, height])
            rect = cv2.boundingRect(np.array(ann_inst['segmentation']))
            x, y, w, h = rect

            # Filter small boxes using argument
            if w < min_crop_size or h < min_crop_size:
                skipped_stats[ann_inst['label']]['too_small'] += 1
                continue

            # Crop
            crop = img[y:y + h, x:x + w]
            
            # Normalize segmentation for the crop
            # Copy to avoid mutating original if referenced elsewhere (though dicts are shared)
            # We modify 'segmentation' and save it.
            crop_seg = [(pt[0] - x, pt[1] - y) for pt in ann_inst['segmentation']]
            # Only update for the record if needed, but FiftyOne sample uses relative coords
            
            ann_id = str(ann_inst.get('id'))
            path_to_save = os.path.join(folder_to_save_files, f"{ann_id}.png")
            
            try:
                cv2.imwrite(path_to_save, crop)
            except Exception as e:
                print(f"\nError saving crop {path_to_save}: {e}")
                skipped_stats[ann_inst['label']]['save_error'] += 1
                continue

            # FiftyOne uses relative coordinates [0, 1]
            new_poly = [(pt[0] / w, pt[1] / h) for pt in crop_seg]
            
            # Update record for JSON
            ann_inst['segmentation'] = crop_seg # Saving crop-relative coordinates in JSON result? 
            # Original script did: ann_inst['segmentation'] = [(pt[0] - x, pt[1] - y) ...]
            records.append(ann_inst)
            
            # Create FiftyOne Sample
            tag_odm = 'odm_true' if ann_inst.get('include_in_odm') else 'odm_false'
            
            sample = fo.Sample(filepath=path_to_save, tags=[tag_odm])
            sample["polyline"] = fo.Polyline(
                label=ann_inst['label'],
                points=[new_poly],
                closed=True,
                filled=False
            )
            
            # Calculation for area and store metadata
            poly_x = [p[0] for p in new_poly]
            poly_y = [p[1] for p in new_poly]
            sample["area"] = polygon_area(poly_x, poly_y)
            sample['width'] = w
            sample['height'] = h
            sample['drawn_fish_id'] = ann_inst.get('fishial_extra', {}).get('drawn_fish_id', None)
            sample["annotation_id"] = ann_inst.get('id')
            sample["image_id"] = str(ann_inst.get('image_id'))
            
            samples.append(sample)
            processed_count += 1

    # Save processed annotation records to JSON
    annotation_save_path = os.path.join(dst_path, "annotation.json")
    save_json(records, annotation_save_path)
    print(f"\nAnnotations saved to {annotation_save_path}")

    # Save skipped statistics
    skipped_save_path = os.path.join(dst_path, "skipped_stats.json")
    save_json(skipped_stats, skipped_save_path)
    print(f"Skipped statistics saved to {skipped_save_path}")

    # Create and save FiftyOne dataset
    if samples:
        dataset = fo.Dataset(voxel_dataset_name)
        dataset.add_samples(samples)
        dataset.persistent = True
        dataset.save()
        print(f"FiftyOne dataset '{voxel_dataset_name}' created with {len(samples)} samples.")
    else:
        print("Warning: No samples were generated.")


def main() -> None:
    """
    Main function to parse arguments and process annotations.
    """
    args = parse_arguments()
    process_annotations(args)


if __name__ == '__main__':
    main()