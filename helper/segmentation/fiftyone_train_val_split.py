"""
This script assigns 'train' and 'val' tags to a FiftyOne dataset's samples.
It provides two modes for splitting the dataset:

1. Random Mode: Assigns a percentage of samples to 'val' and the rest to 'train'.
2. JSON Mode: Reads image IDs from a JSON file and assigns 'val' to those IDs, while others are 'train'.

Usage:
- Random split:
    python script.py --dataset_name my_dataset --mode random --val_percent 0.2
- JSON-based split:
    python script.py --dataset_name my_dataset --mode json --json_path val_ids.json

Arguments:
- dataset_name (str): Name of the FiftyOne dataset.
- mode (str): 'random' for percentage-based splitting, 'json' for a predefined split.
- val_percent (float, optional): Percentage of data to assign to 'val' (only for 'random' mode).
- json_path (str, optional): Path to JSON file containing validation image IDs (only for 'json' mode).

Author: Andrew Ludkiewicz
Date: YYYY-MM-DD
"""

import fiftyone as fo
import fiftyone.core.dataset as fod
import argparse
import json
import random
import logging
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

setup_logging()

def add_tags_random(dataset_name, val_percent):
    dataset = fod.load_dataset(dataset_name)
    
    # Get all sample IDs
    sample_ids = dataset.values("id")

    # Shuffle and split
    random.shuffle(sample_ids)
    split_index = int(val_percent * len(sample_ids))

    val_ids = sample_ids[:split_index]
    train_ids = sample_ids[split_index:]

    # Assign tags
    dataset.select(train_ids).tag_samples("train")
    dataset.select(val_ids).tag_samples("val")
    dataset.save()
    
    logging.info(f"Dataset '{dataset_name}' split: {split_index} val, {len(sample_ids) - split_index} train.")

def add_tags_from_json(dataset_name, json_path):
    dataset = fod.load_dataset(dataset_name)
    
    try:
        with open(json_path, 'r') as f:
            val_ids = set(json.load(f))
        
        for sample in tqdm(dataset, desc="Tagging dataset from JSON"):
            if sample["image_id"] in val_ids:
                sample.tags = ["val"]
            else:
                sample.tags = ["train"]
            sample.save()
        dataset.save()
        logging.info(f"Dataset '{dataset_name}' tagged using provided JSON file.")
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset_name", required=True, help="Name of the FiftyOne dataset")
    parser.add_argument("-m", "--mode", choices=["random", "json"], required=True, help="Tagging mode: 'random' or 'json'")
    parser.add_argument("-p", "--val_percent", type=float, default=0.2, help="Percentage for validation set (only for 'random' mode)")
    parser.add_argument("-j", "--json_path", type=str, help="Path to JSON file containing validation image IDs (only for 'json' mode)")
    
    args = parser.parse_args()
    
    if args.mode == "random":
        add_tags_random(args.dataset_name, args.val_percent)
    elif args.mode == "json":
        if not args.json_path:
            logging.error("JSON path is required for 'json' mode")
        else:
            add_tags_from_json(args.dataset_name, args.json_path)
