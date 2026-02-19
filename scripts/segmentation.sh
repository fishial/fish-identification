#!/bin/bash
# segmentation.sh
# This script executes segmentation tasks:
# 1. Convert COCO annotations to voxel format.
# 2. Split the segmentation dataset.
# 3. Train the segmentation model.
#
# Usage:
#   ./segmentation.sh [options]
#
# Options:
#   -h            Show this help message.
#   -c <file>     COCO annotation file.
#   -i <dir>      Images directory.
#   -d <name>     Segmentation dataset name.
#   -s <dir>      Segmentation save directory.

usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h            Show help"
  echo "  -c <file>     COCO annotation file"
  echo "  -i <dir>      Images directory"
  echo "  -d <name>     Segmentation dataset name"
  echo "  -s <dir>      Segmentation save directory"
  exit 1
}

# Default values
# Coco file json export path
COCO_FILE="/home/fishial/Fishial/dataset/EXPORT_V_0_9/Fishial_Export_Jan_08_2026_04_14_Production_AI_Gen_All_Verified.json"
# directory where images are saved
IMAGES_DIR="/home/fishial/Fishial/dataset/EXPORT_V_0_8/data"

# Fiftyone dataset name
DATASET_SEGMENT="segmentation_dataset_v0.10_with_meta"


# COCO_FILE="/home/fishial/Fishial/dataset/EXPORT_V_0_9/Fishial_Export_Jan_07_2026_05_15_Prod_Export_Test_Images_for_testing.json"
# # directory where images are saved
# IMAGES_DIR="/home/fishial/Fishial/dataset/EXPORT_V_0_8_TEST/data"

# # Fiftyone dataset name
# DATASET_SEGMENT="segmentation_dataset_v0.10_TEST"


# Directore for save segmentation model files
SAVE_DIR_SEGMENTATION="/home/fishial/Fishial/TEST_PIPINE/segmentation_bash"

while getopts "hc:i:d:s:" opt; do
  case "$opt" in
    h)
      usage
      ;;
    c)
      COCO_FILE="$OPTARG"
      ;;
    i)
      IMAGES_DIR="$OPTARG"
      ;;
    d)
      DATASET_SEGMENT="$OPTARG"
      ;;
    s)
      SAVE_DIR_SEGMENTATION="$OPTARG"
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

convert_coco_to_voxel() {
  echo "Converting COCO annotations to voxel format..."
  python ../helper/segmentation/converterCocoToVoxel.py -c "$COCO_FILE" -i "$IMAGES_DIR" -ds "$DATASET_SEGMENT"
}

split_fiftyone_segmentation() {
  echo "Splitting segmentation dataset using FiftyOne..."
  python ../helper/segmentation/fiftyone_train_val_split.py --dataset_name "$DATASET_SEGMENT" --mode random --val_percent 0.2
}

train_segmentation() {
  echo "Training segmentation model..."
  python ../train_scripts/segmentation/train.py --dataset_name "$DATASET_SEGMENT" --num_workers 1 --save_dir "$SAVE_DIR_SEGMENTATION" #--debug 0.01
}

echo "Starting segmentation pipeline..."
convert_coco_to_voxel
# split_fiftyone_segmentation
# train_segmentation
echo "Segmentation pipeline finished."