#!/bin/bash
# classification.sh
# This script executes classification tasks:
# 1. Create the classification dataset.
# 2. Split the dataset into train and validation sets.
# 3. Train a classification model using triplet loss.
# 4. Train a classification model using cross entropy loss.
#
# Usage:
#   ./classification.sh [options]
#
# Options:
#   -h            Show this help message.
#   -p <dir>      Classification images directory.
#   -i <dir>      Classification input directory.
#   -a <file>     Classification annotation file.
#   -n <name>     Classification dataset name.

usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h            Show help"
  echo "  -p <dir>      Classification images directory"
  echo "  -i <dir>      Classification input directory"
  echo "  -a <file>     Classification annotation file"
  echo "  -n <name>     Classification dataset name"
  exit 1
}

# Default base prefix for file paths
DEFAULT_PREFIX="/home/fishial/Fishial/Experiments/classification/v0.10"
PREFIX="${DEFAULT_PREFIX}"

# Default values for classification dataset creating in fiftyone
CLASSIFICATION_IMAGE_DIR="/home/fishial/Fishial/dataset/EXPORT_V_0_8/data"
CLASSIFICATION_INPUT="/home/fishial/Fishial/dataset/EXPORT_V_0_8/data/data"
CLASSIFICATION_ANNOTATIONS="/home/fishial/Fishial/dataset/EXPORT_V_0_9/Fishial_Export_Jan_08_2026_04_14_Production_AI_Gen_All_Verified.json"
CLASSIFICATION_DS_NAME="classification_v0.10"

MIN_EVAL_IMG=3
MAX_PERCENT_EVA_IMG=0.2
MAX_CNT_IMG_PER_CLASS=350
MIN_CNT_IMG=10

# Split out dataset parameters
MIN_PER_CLASS=3
MAX_SAMPLES=120
EVAL_PERCENT=0.25
TRAIN_DS_NAME="classification_v0.10_train"
VALIDATION_DS_NAME="classification_v0.10_validation"

# Classification training defaults for triplet loss
OUTPUT_FOLDER_CLASSIFICATION="$PREFIX/triplet_loss"
CLASSES_PER_BATCH=4
SAMPLES_PER_CLASS=8
EMBEDDINGS=128
BACKBONE="convnext_tiny"
LOSS_NAME="pnploss"
LEARNING_RATE=0.001
MOMENTUM=0.9
WARMUP_STEPS=500
EPOCH=2
EVAL_EPOCH=2
OPT_LEVEL="O1"
FILE_NAME="experiment_01"
DEVICE="cpu"


# Cross entropy training defaults
OUTPUT_FOLDER_CROSS_ENTROPY="$PREFIX/cross_entropy"
CLASSES_PER_BATCH_CE=5
SAMPLES_PER_CLASS_CE=30
LEARNING_RATE_CE=0.03
EPOCH_CE=2
EVAL_EPOCH_CE=2
OPT_LEVEL_CE="O2"
CHECKPOINT="$OUTPUT_FOLDER_CLASSIFICATION/best_model.ckpt"

while getopts "hp:i:a:n:" opt; do
  case "$opt" in
    h)
      usage
      ;;
    p)
      CLASSIFICATION_IMAGE_DIR="$OPTARG"
      ;;
    i)
      CLASSIFICATION_INPUT="$OPTARG"
      ;;
    a)
      CLASSIFICATION_ANNOTATIONS="$OPTARG"
      ;;
    n)
      CLASSIFICATION_DS_NAME="$OPTARG"
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

create_classification_dataset() {
  echo "Creating classification dataset..."
  python ../helper/classification/classification_dataset_creator.py \
    -dp "$CLASSIFICATION_IMAGE_DIR" \
    -i "$CLASSIFICATION_INPUT" \
    -a "$CLASSIFICATION_ANNOTATIONS" \
    -dsn "$CLASSIFICATION_DS_NAME" \
    --min_eval_img "$MIN_EVAL_IMG" \
    --max_percent_eva_img "$MAX_PERCENT_EVA_IMG" \
    --max_cnt_img_per_class "$MAX_CNT_IMG_PER_CLASS" \
    --min_cnt_img "$MIN_CNT_IMG"
}

split_classification_dataset() {
  echo "Splitting classification dataset into train and validation..."
  python ../helper/classification/split_classification_dataset.py \
    --dataset_main "$CLASSIFICATION_DS_NAME" \
    --min_per_class "$MIN_PER_CLASS" \
    --max_samples "$MAX_SAMPLES" \
    --eval_percent "$EVAL_PERCENT" \
    --new_ds_name_train "$TRAIN_DS_NAME" \
    --new_ds_name_validation "$VALIDATION_DS_NAME"
}

train_classification_triplet() {
  echo "Training classification model with triplet loss..."
  python ../train_scripts/classification/auto_train_triplet.py \
    --output_folder "$OUTPUT_FOLDER_CLASSIFICATION" \
    --train_dataset "$TRAIN_DS_NAME" \
    --val_dataset "$VALIDATION_DS_NAME" \
    --classes_per_batch "$CLASSES_PER_BATCH" \
    --samples_per_class "$SAMPLES_PER_CLASS" \
    --embeddings "$EMBEDDINGS" \
    --backbone "$BACKBONE" \
    --loss_name "$LOSS_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --momentum "$MOMENTUM" \
    --warmup_steps "$WARMUP_STEPS" \
    --epoch "$EPOCH" \
    --eval_epochs "$EVAL_EPOCH" \
    --opt_level "$OPT_LEVEL" \
    --device "$DEVICE"
}

train_classification_cross_entropy() {
  echo "Training classification model with cross entropy loss..."
  python ../train_scripts/classification/cross_entropy.py \
    --output_folder "$OUTPUT_FOLDER_CROSS_ENTROPY" \
    --train_dataset "$TRAIN_DS_NAME" \
    --classes_per_batch "$CLASSES_PER_BATCH_CE" \
    --samples_per_class "$SAMPLES_PER_CLASS_CE" \
    --embeddings "$EMBEDDINGS" \
    --backbone "$BACKBONE" \
    --loss_name cross_entropy \
    --learning_rate "$LEARNING_RATE_CE" \
    --momentum "$MOMENTUM" \
    --warmup_steps "$WARMUP_STEPS" \
    --epoch "$EPOCH_CE" \
    --eval_epochs "$EVAL_EPOCH_CE" \
    --opt_level "$OPT_LEVEL_CE" \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT"
}

echo "Starting classification pipeline..."
# create_classification_dataset
split_classification_dataset
# train_classification_triplet
# train_classification_cross_entropy
echo "Classification pipeline finished."