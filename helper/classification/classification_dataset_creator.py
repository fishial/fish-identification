#!/usr/bin/env python3
"""
Script to create a classification dataset from COCO annotations.

This script processes annotations, extracts image crops, normalizes polygon segmentation,
and creates a FiftyOne dataset with polyline annotations. All parameters (paths, dataset name, etc.)
are provided via command-line arguments.
"""

import os
import sys

# Modify sys.path to include the root directory containing 'fish-identification'
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
    print("SETUP: sys.path updated")
    
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import requests
import matplotlib.pyplot as plt

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Import utility functions from classification and segmentation packages
from module.classification_package.src.utils import read_json, save_json
from module.classification_package.src.dataset import FishialDataset
from module.segmentation_package.src.utils import get_mask

# Modify sys.path to include the root directory containing 'fish-identification'
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
    print("SETUP: sys.path updated")


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
    return parser.parse_args()


def get_category_names(data: dict) -> dict:
    """
    Extract valid category names and counts from COCO data for 'General body shape'.

    Args:
        data (dict): COCO formatted data.

    Returns:
        dict: Mapping of category id to a dictionary containing 'name' (supercategory) and 'cnt'.
    """
    category_ids = {}
    for item in data.get('categories', []):
        if item.get('name') == 'General body shape':
            cat_id = item.get('id')
            if cat_id not in category_ids:
                category_ids[cat_id] = {'name': item.get('supercategory'), 'cnt': 1}
            else:
                category_ids[cat_id]['cnt'] += 1
    return category_ids


def get_category_counts(data: dict) -> dict:
    """
    Count annotations per category in COCO data.

    Args:
        data (dict): COCO formatted data.

    Returns:
        dict: Mapping of category id to count of annotations.
    """
    category_cnt = {}
    for ann in data.get('annotations', []):
        cat_id = ann.get('category_id')
        if cat_id is not None:
            category_cnt[cat_id] = category_cnt.get(cat_id, 0) + 1
    return category_cnt


def get_classes_with_min_annotations(category_counts: dict, min_ann: int = 50) -> list:
    """
    Get list of category ids that have at least min_ann annotations.

    Args:
        category_counts (dict): Mapping of category id to annotation count.
        min_ann (int): Minimum required annotations.

    Returns:
        list: List of tuples (category_id, count) meeting the minimum requirement.
    """
    return [(cat_id, cnt) for cat_id, cnt in category_counts.items() if cnt >= min_ann]


def find_image_by_id(image_id: int, data: dict) -> dict:
    """
    Find an image entry in COCO data by its id.

    Args:
        image_id (int): ID of the image.
        data (dict): COCO formatted data.

    Returns:
        dict: Image entry dictionary if found, otherwise None.
    """
    for img in data.get('images', []):
        if img.get('id') == image_id:
            return img
    return None


def get_list_of_files_in_folder(path: str) -> list:
    """
    Get list of file names in the given folder.

    Args:
        path (str): Directory path.

    Returns:
        list: List of file names.
    """
    return next(os.walk(path))[2]


def download_file(download_info: tuple) -> None:
    """
    Download a file from a URL and save it to a specified path.

    Args:
        download_info (tuple): Tuple containing (url, save_path, progress_info).
    """
    url, save_path, progress_info = download_info
    r = requests.get(url, allow_redirects=True)
    with open(save_path, 'wb') as f:
        f.write(r.content)
    print(f"Downloading {progress_info}", end='\r')


def get_image(data: dict, folder_main: str, image_id: int) -> np.ndarray:
    """
    Retrieve an image from disk using its entry in COCO data.

    Args:
        data (dict): COCO formatted data.
        folder_main (str): Folder containing the images.
        image_id (int): ID of the image.

    Returns:
        np.ndarray: Image array read by cv2, or None if not found.
    """
    img_entry = find_image_by_id(image_id, data)
    if img_entry:
        img_path = os.path.join(folder_main, img_entry.get('file_name', ''))
        return cv2.imread(img_path)
    return None


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


def get_annotations_for_image(data_full: dict, img_id: int, valid_category: dict) -> list:
    """
    Get all annotations for a specific image id that belong to valid categories.

    Args:
        data_full (dict): Full COCO data.
        img_id (int): Image ID.
        valid_category (dict): Mapping of valid category ids.

    Returns:
        list: List of annotations for the image.
    """
    annotations = []
    for ann in data_full.get('annotations', []):
        if ann.get('image_id') == img_id and ann.get('category_id') in valid_category:
            annotations.append(ann)
    return annotations


def get_mask_by_annotation(data: dict, ann: dict, main_folder: str, box: bool = False) -> np.ndarray:
    """
    Generate a mask for an annotation either by cropping the bounding box or using a segmentation mask.

    Args:
        data (dict): COCO formatted data.
        ann (dict): Annotation dictionary.
        main_folder (str): Folder containing the images.
        box (bool): If True, return bounding box crop; otherwise, return segmentation mask.

    Returns:
        np.ndarray: Mask image, or None if mask could not be created.
    """
    segmentation = ann.get('segmentation', [])
    if not segmentation:
        return None
    # If segmentation is nested, extract the first element
    seg = segmentation[0] if isinstance(segmentation[0], list) else segmentation

    polygon_points = [(int(seg[i]), int(seg[i + 1])) for i in range(0, len(seg), 2)]
    img = get_image(data, main_folder, ann.get('image_id'))
    if img is None:
        return None

    if box:
        rect = cv2.boundingRect(np.array(polygon_points))
        x, y, w, h = rect
        mask = img[y:y + h, x:x + w].copy()
        if mask.size == 0:
            return None
    else:
        mask = get_mask(img, np.array(polygon_points))
    return mask


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
    Process annotations from COCO JSON, extract valid image annotations, crop images, 
    normalize polygon coordinates, and create a FiftyOne dataset.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    voxel_dataset_name = args.voxel_dataset_name
    dst_path = args.dst_path
    path_to_src_coco_json = args.path_to_src_coco_json
    path_full_images = args.path_to_full_images
    min_cnt_img = args.min_cnt_img

    # List files in the images folder and read JSON data
    list_of_files = get_list_of_files_in_folder(path_full_images)
    data_full = read_json(path_to_src_coco_json)

    # Create destination folder for cropped images
    folder_to_save_files = os.path.join(dst_path, 'images')
    os.makedirs(folder_to_save_files, exist_ok=True)

    # Filter valid images (exist on disk and not flagged as invalid)
    valid_images = {}
    print(f"MAIN FOLDER PATH: {path_full_images}")
    print(f"Total files in folder: {len(list_of_files)}")
    for img in data_full.get('images', []):
        if img.get('is_invalid'):
            continue
        if img.get('file_name') in list_of_files:
            valid_images[img.get('id')] = img
    print(f"Number of valid images: {len(valid_images)}")

    # Get valid categories for processing annotations
    valid_category = get_valid_categories(data_full)
    # Initialize dictionary to collect annotations per category (using supercategory as key)
    annotations_by_category = {cat_name: [] for cat_name in valid_category.values()}

    # Process each annotation
    for ann_idx, ann in enumerate(data_full.get('annotations', [])):
        print(f"Processing annotation {ann_idx}/{len(data_full.get('annotations', []))}", end='\r')
        try:
            if ann.get('image_id') not in valid_images:
                continue
            if ann.get('category_id') not in valid_category:
                continue
        except Exception:
            continue

        # Convert segmentation into list of (x, y) tuples (assuming a single polygon per annotation)
        seg = ann.get('segmentation', [])
        if seg:
            points = [(int(seg[0][i]), int(seg[0][i + 1])) for i in range(0, len(seg[0]), 2)]
            ann['segmentation'] = points
        else:
            continue

        # Add extra info from image metadata
        fishial_extra = valid_images[ann.get('image_id')].get('fishial_extra', {})
        ann['include_in_odm'] = fishial_extra.get('include_in_odm', False)
        # Append annotation under its valid category
        cat_name = valid_category.get(ann.get('category_id'))
        if cat_name:
            annotations_by_category[cat_name].append(ann)

    # Remove categories with fewer than min_cnt_img annotations
    for cat in list(annotations_by_category.keys()):
        if len(annotations_by_category[cat]) < min_cnt_img:
            del annotations_by_category[cat]
    print(f"\nTotal approved categories: {len(annotations_by_category)}")

    # Create label mapping from category name to an integer id
    label_mapping = {label: idx for idx, label in enumerate(annotations_by_category.keys())}
    print(f"Label mapping: {label_mapping}")

    # Combine all annotations (currently only one split is used)
    combined_annotations = []
    for cat, anns in annotations_by_category.items():
        # Sort annotations by 'include_in_odm' flag in descending order
        sorted_anns = sorted(anns, key=lambda d: d.get('include_in_odm', False), reverse=True)
        combined_annotations.extend(sorted_anns)

    # Update annotations with internal id and label based on category
    for ann in combined_annotations:
        cat_super = valid_category.get(ann.get('category_id'))
        ann['id_internal'] = label_mapping.get(cat_super)
        ann['label'] = cat_super

    records = []
    samples = []
    # Process each annotation to create cropped image and annotate
    for idx, ann_inst in enumerate(combined_annotations):
        print(f"Creating sample {idx}/{len(combined_annotations)}", end='\r')

        img_path = os.path.join(path_full_images, valid_images[ann_inst.get('image_id')]['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        # Fix polygon to be within image boundaries
        ann_inst['segmentation'] = fix_polygon(ann_inst['segmentation'], [width, height])
        rect = cv2.boundingRect(np.array(ann_inst['segmentation']))
        x, y, w, h = rect
        # Skip small bounding boxes
        if w < 80 or h < 80:
            continue

        # Crop image using bounding box
        crop = img[y:y + h, x:x + w]
        # Update segmentation coordinates relative to crop
        ann_inst['segmentation'] = [(pt[0] - x, pt[1] - y) for pt in ann_inst['segmentation']]

        ann_id = str(ann_inst.get('id'))
        path_to_save = os.path.join(folder_to_save_files, f"{ann_id}.png")
        try:
            cv2.imwrite(path_to_save, crop)
        except Exception as e:
            print(f"Error saving image {path_to_save}: {e}")
            continue

        # Normalize polygon points to relative coordinates
        new_poly = [(pt[0] / w, pt[1] / h) for pt in ann_inst['segmentation']]
        tag_odm = 'odm_true' if ann_inst.get('include_in_odm') else 'odm_false'
        records.append(ann_inst)

        sample = fo.Sample(filepath=path_to_save, tags=[tag_odm])
        sample["polyline"] = fo.Polyline(
            label=ann_inst['label'],
            points=[new_poly],
            closed=True,
            filled=False
        )
        sample["area"] = polygon_area([p[0] for p in new_poly], [p[1] for p in new_poly])
        sample['width'] = w
        sample['height'] = h
        sample['drawn_fish_id'] = ann_inst.get('fishial_extra', {}).get('drawn_fish_id', None)
        sample["annotation_id"] = ann_inst.get('id')
        sample["image_id"] = str(ann_inst.get('image_id'))
        samples.append(sample)

    # Save processed annotation records to JSON
    annotation_save_path = os.path.join(dst_path, "annotation.json")
    save_json(records, annotation_save_path)
    print(f"\nAnnotations saved to {annotation_save_path}")

    # Create and save FiftyOne dataset
    dataset = fo.Dataset(voxel_dataset_name)
    dataset.add_samples(samples)
    dataset.persistent = True
    dataset.save()
    print(f"FiftyOne dataset '{voxel_dataset_name}' created with {len(samples)} samples.")


def main() -> None:
    """
    Main function to parse arguments and process annotations.
    """
    args = parse_arguments()
    process_annotations(args)


if __name__ == '__main__':
    main()