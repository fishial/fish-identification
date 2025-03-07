#!/bin/bash
# object_detection.sh
# This script executes object detection tasks:
# 1. Convert FiftyOne dataset to YOLO format for fish detection.
# 2. Train the object detection model.
#
# Usage:
#   ./object_detection.sh [options]
#
# Options:
#   -h         Show help.
#   -x <prefix> Base prefix for file paths.
#   -d <name>  Dataset name for fish detection.
#   -o <dir>   Output directory for YOLO dataset.
#   -n <num>   Number of classes (default: 1).
#   -y <yaml>  Data YAML file for object detection.
#   -p <proj>  Project directory for object detection.
#   -r <name>  Run name for object detection training.

usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h         Show help"
  echo "  -x <prefix> Base prefix for file paths"
  echo "  -d <name>  Dataset name for fish detection"
  echo "  -o <dir>   Output directory for YOLO dataset"
  echo "  -n <num>   Number of classes (default: 1)"
  echo "  -y <yaml>  Data YAML file for object detection"
  echo "  -p <proj>  Project directory for object detection"
  echo "  -r <name>  Run name for object detection training"
  exit 1
}

# Default base prefix for file paths
DEFAULT_PREFIX="/home/fishial/Fishial/TEST_PIPINE/fish_detection_bash_2"
PREFIX="${DEFAULT_PREFIX}"

# Default values for creating YOLO format dataset and training configuration
FO_FISH_DETECTION_DATASET="test_segment_bash_2"
NUM_CLASSES=1
# Output directory for YOLO dataset; will be set later using PREFIX if not provided
COCO_FISH_DETECTION_OUTPUT_DIR="$DEFAULT_PREFIX/yolo_dataset"

# Model and training parameters
MODEL="yolov10s.yaml"                            # YOLO model config file
DATA="$COCO_FISH_DETECTION_OUTPUT_DIR/data.yaml" # Data YAML file; will be set later using PREFIX if not provided
EPOCHS=10                                        # Number of training epochs
BATCH=8                                          # Batch size
IMGSZ=640                                        # Input image size (square)
FRACTION=1.0                                     # Fraction of dataset to use
PATIENCE=150                                     # Early stopping patience (0 to disable)
PRETRAINED="--pretrained"                        # Use this flag if a pretrained model is desired; leave empty if not
PROJECT="$DEFAULT_PREFIX"                        # Project directory; will be set later using PREFIX if not provided
RUN_NAME="my_training_run"                       # Name for the training run (required)
# Uncomment the next line to run in dry-run mode (configuration check only)
# DRY_RUN="--dry_run"



while getopts "hx:d:o:n:y:p:r:" opt; do
  case "$opt" in
    h)
      usage
      ;;
    x)
      PREFIX="$OPTARG"
      ;;
    d)
      FO_FISH_DETECTION_DATASET="$OPTARG"
      ;;
    o)
      COCO_FISH_DETECTION_OUTPUT_DIR="$OPTARG"
      ;;
    n)
      NUM_CLASSES="$OPTARG"
      ;;
    y)
      DATA="$OPTARG"
      ;;
    p)
      PROJECT="$OPTARG"
      ;;
    r)
      RUN_NAME="$OPTARG"
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

# Set default paths using the PREFIX variable if not provided via command-line options
if [ -z "$COCO_FISH_DETECTION_OUTPUT_DIR" ]; then
    COCO_FISH_DETECTION_OUTPUT_DIR="${PREFIX}/dataset"
fi

if [ -z "$PROJECT" ]; then
    PROJECT="${PREFIX}/fish_detection"
fi

if [ -z "$DATA" ]; then
    DATA="${PREFIX}/dataset.yaml"
fi

fish_detection_to_yolo() {
  echo "Converting FiftyOne dataset to YOLO format for fish detection..."
  python ../module/fish_detection/fiftyone_to_yolo.py --dataset "$FO_FISH_DETECTION_DATASET" --output_dir "$COCO_FISH_DETECTION_OUTPUT_DIR" --split_train_val --num_classes "$NUM_CLASSES"
}

train_object_detection() {
  echo "Training object detection model..."
  python ../train_scripts/object_detection/train.py \
    --model "$MODEL" \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    --fraction "$FRACTION" \
    --patience "$PATIENCE" \
    $PRETRAINED \
    --project "$PROJECT" \
    --run_name "$RUN_NAME" \
    $DRY_RUN
}

echo "Starting object detection pipeline..."
fish_detection_to_yolo
train_object_detection
echo "Object detection pipeline finished."