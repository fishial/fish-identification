#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import yaml
import cv2
import numpy as np
import logging
import argparse
import fiftyone as fo
from tqdm import tqdm

# ============== CONFIGURATION ==============
DEFAULT_CLASSES = ["Fish"]  # Modify if multiple classes exist

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============== FUNCTIONS ==============

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare FiftyOne dataset for YOLO training.")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the FiftyOne dataset to process.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the YOLO-formatted dataset.")
    parser.add_argument("--split_train_val", action="store_true",
                        help="Split dataset into train/val based on FiftyOne tags.")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="Number of object classes in the dataset.")
    
    return parser.parse_args()


def create_yaml(output_yaml_path, train_dir, val_dir, num_classes, class_names):
    """Creates a YOLO dataset YAML file."""
    yaml_data = {
        "names": class_names,
        "nc": num_classes,
        "train": train_dir,
        "val": val_dir,
        "test": " "
    }

    with open(output_yaml_path, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
    logging.info(f"Created YAML dataset config at: {output_yaml_path}")


def prepare_dataset(fo_dataset, output_dir, split_train_val, num_classes, class_names):
    """Converts FiftyOne dataset into YOLO format (images + labels)."""
    # Define paths
    train_images_dir = os.path.join(output_dir, "train", "images")
    val_images_dir = os.path.join(output_dir, "val", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_labels_dir = os.path.join(output_dir, "val", "labels")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Create YOLO dataset config file
    create_yaml(os.path.join(output_dir, "data.yaml"), "train/images", "val/images", num_classes, class_names)

    logging.info(f"Preparing dataset in {output_dir}...")

    # Process dataset
    for sample in tqdm(fo_dataset, desc="Processing images"):
        if not sample["General body shape"].polylines:
            continue  # Skip images without polylines

        is_val = "val" in sample.tags if split_train_val else False
        filename = sample.filename
        imgs_dir, label_dir = (val_images_dir, val_labels_dir) if is_val else (train_images_dir, train_labels_dir)

        # Copy image to the correct folder
        shutil.copy(sample.filepath, imgs_dir)

        # Generate YOLO label file
        label_file = os.path.join(label_dir, f"{os.path.splitext(filename)[0]}.txt")
        lines = []

        for polyline in sample["General body shape"].polylines:
            polygon = np.array(polyline.points[0])

            # Convert polygon to bounding box (YOLO format: class xc yc w h)
            x1, y1 = polygon[:, 0].min(), polygon[:, 1].min()
            x2, y2 = polygon[:, 0].max(), polygon[:, 1].max()
            w, h = x2 - x1, y2 - y1
            xc, yc = x1 + w / 2, y1 + h / 2

            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")  # YOLO format

        # Save labels
        with open(label_file, "w") as f:
            f.write("\n".join(lines))

    logging.info("Dataset preparation complete!")


# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    args = parse_arguments()

    logging.info(f"Loading FiftyOne dataset: {args.dataset}")
    fo_dataset = fo.load_dataset(args.dataset)

    # Prepare dataset
    prepare_dataset(fo_dataset, args.output_dir, args.split_train_val, args.num_classes, DEFAULT_CLASSES)

    logging.info(f"Dataset prepared successfully! YOLO dataset saved at: {args.output_dir}")