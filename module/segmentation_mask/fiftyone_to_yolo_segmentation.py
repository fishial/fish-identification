#!/usr/bin/env python
# coding: utf-8

"""
FiftyOne to YOLO Segmentation Dataset Converter
================================================

This script converts FiftyOne datasets with polyline annotations
to YOLO segmentation format (polygon masks).

Features:
  ✅ Converts polylines to YOLO segmentation format
  ✅ Supports train/val split based on FiftyOne tags
  ✅ Automatic YAML dataset configuration
  ✅ Multi-class support
  ✅ Progress tracking with tqdm

Usage:
  python fiftyone_to_yolo_segmentation.py --dataset <dataset_name> --output_dir <output_path>
"""

import os
import shutil
import yaml
import cv2
import numpy as np
import logging
import argparse
import fiftyone as fo
from tqdm import tqdm
from pathlib import Path

# ============== CONFIGURATION ==============
DEFAULT_CLASSES = ["Fish"]  # Modify if multiple classes exist

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ============== FUNCTIONS ==============

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert FiftyOne dataset to YOLO segmentation format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with train/val split
  python fiftyone_to_yolo_segmentation.py --dataset fish_dataset --output_dir ./yolo_seg

  # Single-class fish segmentation
  python fiftyone_to_yolo_segmentation.py --dataset fish_dataset --output_dir ./yolo_seg --num_classes 1

  # Multi-class segmentation
  python fiftyone_to_yolo_segmentation.py --dataset fish_dataset --output_dir ./yolo_seg --num_classes 3 --class_names Fish Shark Dolphin
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the FiftyOne dataset to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the YOLO segmentation dataset"
    )
    parser.add_argument(
        "--split_train_val",
        action="store_true",
        help="Split dataset into train/val based on FiftyOne tags"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of object classes in the dataset (default: 1)"
    )
    parser.add_argument(
        "--class_names",
        nargs='+',
        default=DEFAULT_CLASSES,
        help="Names of the classes (space-separated)"
    )
    parser.add_argument(
        "--polyline_field",
        type=str,
        default="General body shape",
        help="Name of the polyline field in FiftyOne dataset (default: 'General body shape')"
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=3,
        help="Minimum number of points in polygon (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def create_yaml(output_yaml_path, train_dir, val_dir, num_classes, class_names):
    """
    Creates a YOLO segmentation dataset YAML file.
    
    Args:
        output_yaml_path (str): Path to save the YAML file
        train_dir (str): Relative path to training images
        val_dir (str): Relative path to validation images
        num_classes (int): Number of classes
        class_names (list): List of class names
    """
    # Ensure class_names matches num_classes
    if len(class_names) < num_classes:
        class_names = class_names + [f"class_{i}" for i in range(len(class_names), num_classes)]
    elif len(class_names) > num_classes:
        class_names = class_names[:num_classes]
    
    yaml_data = {
        "path": os.path.dirname(output_yaml_path),  # Dataset root dir
        "train": train_dir,
        "val": val_dir,
        "test": "",  # Optional
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": num_classes,
    }

    with open(output_yaml_path, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)
    
    logging.info(f"✓ Created YAML dataset config at: {output_yaml_path}")
    logging.info(f"✓ Classes: {class_names}")


def polygon_to_yolo_format(polygon, img_width, img_height):
    """
    Convert polygon coordinates to YOLO segmentation format.
    
    YOLO segmentation format:
    class_id x1 y1 x2 y2 x3 y3 ... (normalized 0-1)
    
    Args:
        polygon (np.array): Polygon points in format [[x1, y1], [x2, y2], ...]
        img_width (int): Image width
        img_height (int): Image height
    
    Returns:
        str: YOLO formatted polygon string
    """
    # Normalize coordinates to 0-1 range
    normalized_points = []
    for point in polygon:
        x = max(0, min(1, point[0]))  # Clamp to [0, 1]
        y = max(0, min(1, point[1]))  # Clamp to [0, 1]
        normalized_points.extend([x, y])
    
    return " ".join([f"{coord:.6f}" for coord in normalized_points])


def prepare_dataset(fo_dataset, output_dir, split_train_val, num_classes, class_names, polyline_field, min_points, verbose):
    """
    Converts FiftyOne dataset with polylines into YOLO segmentation format.
    
    Args:
        fo_dataset: FiftyOne dataset
        output_dir (str): Output directory path
        split_train_val (bool): Whether to split train/val
        num_classes (int): Number of classes
        class_names (list): List of class names
        polyline_field (str): Name of the polyline field
        min_points (int): Minimum number of points in polygon
        verbose (bool): Verbose output
    """
    # Define paths
    train_images_dir = os.path.join(output_dir, "train", "images")
    val_images_dir = os.path.join(output_dir, "val", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_labels_dir = os.path.join(output_dir, "val", "labels")

    # Create directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Create YOLO dataset config file
    create_yaml(
        os.path.join(output_dir, "data.yaml"),
        "train/images",
        "val/images",
        num_classes,
        class_names
    )

    logging.info("=" * 80)
    logging.info("🔄 Converting FiftyOne dataset to YOLO segmentation format...")
    logging.info("=" * 80)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Polyline field: {polyline_field}")
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Split train/val: {split_train_val}")
    logging.info("=" * 80)

    # Statistics
    stats = {
        'total_samples': 0,
        'train_samples': 0,
        'val_samples': 0,
        'total_polygons': 0,
        'skipped_samples': 0,
        'skipped_polygons': 0,
    }

    # Process dataset
    for sample in tqdm(fo_dataset, desc="Processing images"):
        stats['total_samples'] += 1
        
        # Check if polyline field exists
        if polyline_field not in sample or not sample[polyline_field]:
            if verbose:
                logging.debug(f"Skipping {sample.filename}: no {polyline_field} field")
            stats['skipped_samples'] += 1
            continue
        
        # Get polylines
        polylines = sample[polyline_field].polylines
        if not polylines:
            if verbose:
                logging.debug(f"Skipping {sample.filename}: no polylines")
            stats['skipped_samples'] += 1
            continue

        # Determine if validation sample
        is_val = "val" in sample.tags if split_train_val else False
        
        # Set paths based on split
        imgs_dir = val_images_dir if is_val else train_images_dir
        label_dir = val_labels_dir if is_val else train_labels_dir
        
        # Update stats
        if is_val:
            stats['val_samples'] += 1
        else:
            stats['train_samples'] += 1

        # Get image info
        filename = sample.filename
        filepath = sample.filepath
        
        # Get image dimensions
        if sample.metadata:
            img_width = sample.metadata.width
            img_height = sample.metadata.height
        else:
            # Load image to get dimensions
            img = cv2.imread(filepath)
            if img is None:
                logging.warning(f"Failed to load image: {filepath}")
                stats['skipped_samples'] += 1
                continue
            img_height, img_width = img.shape[:2]

        # Copy image to the correct folder
        try:
            shutil.copy(filepath, imgs_dir)
        except Exception as e:
            logging.warning(f"Failed to copy image {filepath}: {e}")
            stats['skipped_samples'] += 1
            continue

        # Generate YOLO label file
        label_file = os.path.join(label_dir, f"{os.path.splitext(filename)[0]}.txt")
        lines = []

        # Process each polyline
        for polyline in polylines:
            # Get polygon points
            if not polyline.points or len(polyline.points) == 0:
                if verbose:
                    logging.debug(f"Skipping empty polyline in {filename}")
                stats['skipped_polygons'] += 1
                continue
            
            # Get first polygon (YOLO segmentation uses single polygons)
            polygon = np.array(polyline.points[0])
            
            # Validate polygon
            if len(polygon) < min_points:
                if verbose:
                    logging.debug(f"Skipping polygon with {len(polygon)} points (min: {min_points}) in {filename}")
                stats['skipped_polygons'] += 1
                continue
            
            # Get class ID (default to 0 for single class)
            class_id = 0
            if hasattr(polyline, 'label') and polyline.label:
                try:
                    class_id = class_names.index(polyline.label)
                except ValueError:
                    class_id = 0
            
            # Convert to YOLO format
            yolo_polygon = polygon_to_yolo_format(polygon, img_width, img_height)
            
            # Create YOLO segmentation line: class_id x1 y1 x2 y2 ...
            lines.append(f"{class_id} {yolo_polygon}")
            stats['total_polygons'] += 1

        # Save labels (only if we have polygons)
        if lines:
            try:
                with open(label_file, "w") as f:
                    f.write("\n".join(lines))
            except Exception as e:
                logging.warning(f"Failed to write label file {label_file}: {e}")
                continue
        else:
            if verbose:
                logging.debug(f"No valid polygons for {filename}")

    # Print statistics
    logging.info("=" * 80)
    logging.info("✅ CONVERSION COMPLETED")
    logging.info("=" * 80)
    logging.info("Statistics:")
    logging.info(f"  Total samples processed: {stats['total_samples']}")
    logging.info(f"  Training samples: {stats['train_samples']}")
    logging.info(f"  Validation samples: {stats['val_samples']}")
    logging.info(f"  Total polygons converted: {stats['total_polygons']}")
    logging.info(f"  Skipped samples: {stats['skipped_samples']}")
    logging.info(f"  Skipped polygons: {stats['skipped_polygons']}")
    logging.info("=" * 80)
    logging.info(f"Dataset saved to: {output_dir}")
    logging.info(f"YAML config: {os.path.join(output_dir, 'data.yaml')}")
    logging.info("=" * 80)


# ============== MAIN EXECUTION ==============
def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info("=" * 80)
    logging.info("🐟 FiftyOne to YOLO Segmentation Converter")
    logging.info("=" * 80)
    logging.info(f"Loading FiftyOne dataset: {args.dataset}")
    
    try:
        fo_dataset = fo.load_dataset(args.dataset)
        logging.info(f"✓ Dataset loaded: {len(fo_dataset)} samples")
    except Exception as e:
        logging.error(f"❌ Failed to load dataset: {e}")
        return 1

    # Prepare dataset
    try:
        prepare_dataset(
            fo_dataset,
            args.output_dir,
            args.split_train_val,
            args.num_classes,
            args.class_names,
            args.polyline_field,
            args.min_points,
            args.verbose
        )
        
        logging.info("🎉 Conversion completed successfully!")
        logging.info("")
        logging.info("Next steps:")
        logging.info(f"  1. Review dataset: ls {args.output_dir}")
        logging.info(f"  2. Train model: python train_yolo.py --data {os.path.join(args.output_dir, 'data.yaml')}")
        logging.info("")
        
        return 0
        
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
