#!/usr/bin/env python3
"""
Script to split a FiftyOne classification dataset into train and validation splits.

For each label in the "polyline.label" field, the script:
  - Filters samples with the current label.
  - Separates samples based on the tags "odm_true" and "odm_false".
  - Shuffles the sample IDs and limits the number per class to a maximum.
  - Uses a specified percentage of samples for validation (if there are enough samples).
  - If a class has fewer than the minimum required samples, all samples are assigned to validation.

New datasets for train and validation are created and saved.
"""

import random
import argparse
from tqdm import tqdm
import fiftyone as fo


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset splitting parameters.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Split a FiftyOne classification dataset into train and validation splits."
    )
    parser.add_argument("--dataset_main", type=str, default="classification-v0.8.1",
                        help="Name of the main dataset to load")
    parser.add_argument("--min_per_class", type=int, default=40,
                        help="Minimum number of samples required per class")
    parser.add_argument("--max_samples", type=int, default=250,
                        help="Maximum number of samples per class to consider")
    parser.add_argument("--eval_percent", type=float, default=0.25,
                        help="Percentage of samples to use for validation")
    parser.add_argument("--new_ds_name_train", type=str, default=None,
                        help="Name for the new train dataset (auto-generated if not provided)")
    parser.add_argument("--new_ds_name_validation", type=str, default=None,
                        help="Name for the new validation dataset (auto-generated if not provided)")
    return parser.parse_args()


def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()

    # Auto-generate new dataset names if not provided
    if args.new_ds_name_train is None:
        args.new_ds_name_train = f"{args.dataset_main}_{args.min_per_class}_{args.max_samples}_TRAIN"
    if args.new_ds_name_validation is None:
        args.new_ds_name_validation = f"{args.dataset_main}_{args.min_per_class}_{args.max_samples}_VALIDATION"

    # Load the main dataset
    ds = fo.load_dataset(args.dataset_main)
    
    # Create new datasets for train and validation splits
    new_dataset_train = fo.Dataset(args.new_ds_name_train)
    new_dataset_validation = fo.Dataset(args.new_ds_name_validation)

    # Retrieve distinct labels from the "polyline.label" field
    all_labels = ds.distinct("polyline.label")

    for label in tqdm(all_labels, desc="Processing labels"):
        # Filter samples that have the current label in the "polyline" field
        view = ds.filter_labels("polyline", fo.ViewField("label") == label)
        view_odm_true_id = view.match_tags(["odm_true"]).values("id")
        view_odm_false_id = view.match_tags(["odm_false"]).values("id")

        # Shuffle the sample IDs for randomness
        random.shuffle(view_odm_true_id)
        random.shuffle(view_odm_false_id)

        # Combine both lists
        all_class_ids = view_odm_true_id + view_odm_false_id

        if len(all_class_ids) >= args.min_per_class:
            num_selected = min(args.max_samples, len(all_class_ids))
            val_samples = int(num_selected * args.eval_percent)
            train_samples = num_selected - val_samples
            rest_samples = max(0, len(all_class_ids) - args.max_samples)

            # Assign the first MAX_SAMPLES samples to the new train dataset,
            # tagging a portion as "val" based on the eval percentage.
            for idx, sample_id in enumerate(all_class_ids[:args.max_samples]):
                sample = ds[sample_id]
                tag = "val" if idx < val_samples else "train"
                if hasattr(sample, "tags"):
                    sample.tags.append(tag)
                else:
                    sample["tags"] = [tag]
                new_dataset_train.add_sample(sample)

            # Any remaining samples are added to the validation dataset.
            for idx, sample_id in enumerate(all_class_ids[args.max_samples:]):
                sample = ds[sample_id]
                if hasattr(sample, "tags"):
                    sample.tags.append("train")
                else:
                    sample["tags"] = ["train"]
                new_dataset_validation.add_sample(sample)

            print(f"Label: {label} | train: {train_samples} | val: {val_samples} | rest: {rest_samples}")
        else:
            # For classes with fewer than the minimum samples, assign all to validation.
            for sample_id in all_class_ids:
                sample = ds[sample_id]
                new_dataset_validation.add_sample(sample)
            print(f"Label: {label} | Validation only: {len(all_class_ids)}")

    new_dataset_train.save()
    new_dataset_train.persistent = True
    
    new_dataset_validation.save()
    new_dataset_validation.persistent = True
    print("Train and validation datasets saved successfully.")


if __name__ == "__main__":
    main()