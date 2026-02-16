#!/bin/bash
################################################################################
# 🐟 Advanced Segmentation Pipeline (YOLO-based)
# 
# This script executes optimized segmentation tasks:
# 1. Convert FiftyOne dataset to YOLO segmentation format (auto-skip if exists)
# 2. Train the segmentation model with advanced optimization
#
# Features:
#   ✅ Simple, explicit configuration
#   ✅ Resume training from checkpoint
#   ✅ Optimized for segmenting multiple fish
#   ✅ Advanced augmentation settings
#   ✅ Automatic dataset detection (no recreation if exists)
#   ✅ GPU detection and configuration
#
# Usage:
#   ./segmentation_yolo.sh [options]
#
# Options:
#   -h           Show help
#   -d <name>    Dataset name in FiftyOne
#   -o <dir>     Output directory for YOLO dataset
#   -n <num>     Number of classes (default: 1)
#   -m <path>    Model path (default: yolo26n-seg.pt)
#   -c <path>    Resume from checkpoint (path to last.pt or best.pt)
#   -p <proj>    Project directory for training output
#   -r <name>    Run name for this training session
#   -e <epochs>  Number of epochs (default: 300)
#   -i <size>    Image size (default: 640)
#   -g <gpu>     GPU device (default: 0, use 'cpu' for CPU)
#   -v           Verbose mode (show all details)
#   -q           Quiet mode (minimal output)
#   -t           Dry run (test configuration only)
################################################################################

# ============== COLORS ==============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

usage() {
  echo -e "${CYAN}${BOLD}Usage:${NC} $0 [options]"
  echo ""
  echo -e "${CYAN}Options:${NC}"
  echo "  -h              Show this help message"
  echo "  -d <name>       Dataset name in FiftyOne"
  echo "  -o <dir>        Output directory for YOLO dataset"
  echo "  -n <num>        Number of classes (default: 1)"
  echo "  -m <path>       Model path (default: yolo26n-seg.pt)"
  echo "  -c <path>       Resume from checkpoint (path to last.pt or best.pt)"
  echo "  -p <proj>       Project directory for training output"
  echo "  -r <name>       Run name for this training session"
  echo "  -e <epochs>     Number of epochs (default: 300)"
  echo "  -i <size>       Image size (default: 640)"
  echo "  -g <gpu>        GPU device (default: 0, use 'cpu' for CPU)"
  echo "  -v              Verbose mode (show all training details)"
  echo "  -q              Quiet mode (minimal output)"
  echo "  -t              Dry run (test configuration only)"
  echo ""
  echo -e "${CYAN}Examples:${NC}"
  echo ""
  echo "  ${BOLD}# New training${NC}"
  echo "  $0 -d my_dataset -r fish_seg_v1"
  echo ""
  echo "  ${BOLD}# Resume from checkpoint${NC}"
  echo "  $0 -d my_dataset -c ./runs/fish_seg_v1/weights/last.pt -r fish_seg_v1_resumed"
  echo ""
  echo "  ${BOLD}# Custom model and image size${NC}"
  echo "  $0 -d my_dataset -m yolo26m-seg.pt -i 1280 -e 500 -r fish_seg_v2"
  echo ""
  echo "  ${BOLD}# Dry run to test configuration${NC}"
  echo "  $0 -d my_dataset -r test -t"
  echo ""
  echo -e "${CYAN}Notes:${NC}"
  echo "  • Dataset conversion will be automatically skipped if YOLO dataset already exists"
  echo "  • When resuming, model (-m) will be ignored and loaded from checkpoint"
  echo "  • Checkpoints typically located at: <project>/<run_name>/weights/last.pt"
  echo ""
  exit 1
}

# ============== DEFAULT CONFIGURATION ==============

# Dataset configuration
FO_FISH_SEGMENTATION_DATASET="segmentation_merged_v0.1_full"
NUM_CLASSES=1
YOLO_FISH_SEGMENTATION_OUTPUT_DIR="/home/fishial/Fishial/Experiments/v10/segmentation/segmentation_merged_v0_1_full"

# Model configuration
MODEL="/home/fishial/Fishial/Experiments/v10/segmentation/fish_segmentation_20260207_162332/weights/best.pt"
RESUME_CHECKPOINT=""  # Path to checkpoint for resume training

# Project configuration
PROJECT="/home/fishial/Fishial/Experiments/v10/segmentation"
RUN_NAME="fish_segmentation_$(date +%Y%m%d_%H%M%S)"

# GPU configuration
GPU_DEVICE="0"

# Training parameters
EPOCHS=300
BATCH=-1  # Auto batch size
IMGSZ=640
PATIENCE=100
WORKERS=8

# Detection/Segmentation parameters (optimized for multiple fish)
CONF=0.15
IOU=0.45
MAX_DET=500

# Augmentation parameters
AUGMENT_LEVEL="heavy"
MOSAIC=1.0
MIXUP=0.15
COPY_PASTE=0.3

# Optimization parameters
OPTIMIZER="AdamW"
LR0=0.001
AMP=true
MULTI_SCALE=false  # Disabled to avoid interpolation errors
SINGLE_CLS=true

# Segmentation-specific settings
OVERLAP_MASK=true
MASK_RATIO=4
RETINA_MASKS=true

# Polyline field in FiftyOne dataset
POLYLINE_FIELD="General body shape"

# Flags
DRY_RUN=false
VERBOSE=false
QUIET=false


# ============== PARSE ARGUMENTS ==============
while getopts "hd:o:n:m:c:p:r:e:i:g:vqt" opt; do
  case "$opt" in
    h)
      usage
      ;;
    d)
      FO_FISH_SEGMENTATION_DATASET="$OPTARG"
      ;;
    o)
      YOLO_FISH_SEGMENTATION_OUTPUT_DIR="$OPTARG"
      ;;
    n)
      NUM_CLASSES="$OPTARG"
      ;;
    m)
      MODEL="$OPTARG"
      ;;
    c)
      RESUME_CHECKPOINT="$OPTARG"
      ;;
    p)
      PROJECT="$OPTARG"
      ;;
    r)
      RUN_NAME="$OPTARG"
      ;;
    e)
      EPOCHS="$OPTARG"
      ;;
    i)
      IMGSZ="$OPTARG"
      ;;
    g)
      GPU_DEVICE="$OPTARG"
      ;;
    v)
      VERBOSE=true
      ;;
    q)
      QUIET=true
      ;;
    t)
      DRY_RUN=true
      ;;
    *)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

# Set data.yaml path based on output directory
DATA="$YOLO_FISH_SEGMENTATION_OUTPUT_DIR/data.yaml"

# ============== HELPER FUNCTIONS ==============

print_header() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================================"
    echo "  🐟 ADVANCED FISH SEGMENTATION PIPELINE (YOLO)"
    echo "================================================================================"
    echo -e "${NC}"
}

print_section() {
    echo -e "${BLUE}${BOLD}$1${NC}"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
        return 0
    else
        echo -e "${YELLOW}⚠ No GPU detected - training will use CPU${NC}"
        GPU_DEVICE="cpu"
        return 1
    fi
}

validate_checkpoint() {
    local checkpoint="$1"
    
    if [ -z "$checkpoint" ]; then
        return 0  # No checkpoint specified, skip validation
    fi
    
    if [ ! -f "$checkpoint" ]; then
        echo -e "${RED}❌ Checkpoint file not found: $checkpoint${NC}"
        return 1
    fi
    
    # Check if file has .pt extension
    if [[ ! "$checkpoint" =~ \.pt$ ]]; then
        echo -e "${RED}❌ Checkpoint must be a .pt file${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Checkpoint validated: $checkpoint${NC}"
    return 0
}

validate_model() {
    local model="$1"
    
    if [ -z "$model" ]; then
        echo -e "${RED}❌ Model path is required${NC}"
        return 1
    fi
    
    # If it's a standard model name (like yolo26n-seg.pt), check in saved_models directory
    if [[ ! "$model" =~ ^/ ]]; then
        local model_dir="/home/fishial/Fishial/saved_models"
        if [ -f "$model_dir/$model" ]; then
            MODEL="$model_dir/$model"
            echo -e "${GREEN}✓ Model found: $MODEL${NC}"
            return 0
        fi
    fi
    
    # Check if file exists
    if [ ! -f "$model" ]; then
        echo -e "${RED}❌ Model file not found: $model${NC}"
        echo -e "${YELLOW}Tip: Use full path or place model in /home/fishial/Fishial/saved_models/${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Model validated: $model${NC}"
    return 0
}

save_config() {
    local config_file="$1"
    
    if [ "$DRY_RUN" = true ]; then
        return 0
    fi
    
    cat > "$config_file" <<EOF
# Segmentation Training Configuration - $(date)
# Generated by segmentation_yolo.sh

[Dataset]
FiftyOne_Dataset = $FO_FISH_SEGMENTATION_DATASET
YOLO_Output_Dir = $YOLO_FISH_SEGMENTATION_OUTPUT_DIR
Data_YAML = $DATA
Num_Classes = $NUM_CLASSES
Polyline_Field = $POLYLINE_FIELD

[Model]
Model = $MODEL
Resume_Checkpoint = ${RESUME_CHECKPOINT:-None}

[Training]
Project = $PROJECT
Run_Name = $RUN_NAME
Epochs = $EPOCHS
Batch = $BATCH
Image_Size = $IMGSZ
Patience = $PATIENCE
Workers = $WORKERS
GPU_Device = $GPU_DEVICE

[Segmentation]
Confidence = $CONF
IoU_Threshold = $IOU
Max_Detections = $MAX_DET
Overlap_Mask = $OVERLAP_MASK
Mask_Ratio = $MASK_RATIO
Retina_Masks = $RETINA_MASKS

[Augmentation]
Level = $AUGMENT_LEVEL
Mosaic = $MOSAIC
MixUp = $MIXUP
Copy_Paste = $COPY_PASTE
Multi_Scale = $MULTI_SCALE

[Optimization]
Optimizer = $OPTIMIZER
Learning_Rate = $LR0
AMP = $AMP
Single_Class = $SINGLE_CLS
EOF
    
    echo -e "${GREEN}✓ Configuration saved to: $config_file${NC}"
}

# ============== PIPELINE FUNCTIONS ==============

check_dataset_exists() {
    local dataset_dir="$1"
    
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}[DEBUG] Checking dataset at: $dataset_dir${NC}"
    fi
    
    # Check if dataset directory exists
    if [ ! -d "$dataset_dir" ]; then
        if [ "$VERBOSE" = true ]; then
            echo -e "${YELLOW}[DEBUG] Dataset directory does not exist${NC}"
        fi
        return 1
    fi
    
    # Check if data.yaml exists
    if [ ! -f "$dataset_dir/data.yaml" ]; then
        if [ "$VERBOSE" = true ]; then
            echo -e "${YELLOW}[DEBUG] data.yaml not found${NC}"
        fi
        return 1
    fi
    
    # Check for both possible directory structures
    # Structure 1: train/images and train/labels (used by fiftyone_to_yolo_segmentation.py)
    # Structure 2: images/train and labels/train (alternative YOLO format)
    local train_images_dir=""
    local train_labels_dir=""
    
    if [ -d "$dataset_dir/train/images" ] && [ -d "$dataset_dir/train/labels" ]; then
        train_images_dir="$dataset_dir/train/images"
        train_labels_dir="$dataset_dir/train/labels"
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}[DEBUG] Found structure: train/images, train/labels${NC}"
        fi
    elif [ -d "$dataset_dir/images/train" ] && [ -d "$dataset_dir/labels/train" ]; then
        train_images_dir="$dataset_dir/images/train"
        train_labels_dir="$dataset_dir/labels/train"
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}[DEBUG] Found structure: images/train, labels/train${NC}"
        fi
    else
        if [ "$VERBOSE" = true ]; then
            echo -e "${YELLOW}[DEBUG] Train directories missing${NC}"
            echo -e "${YELLOW}[DEBUG] train/images exists: $([ -d "$dataset_dir/train/images" ] && echo "yes" || echo "no")${NC}"
            echo -e "${YELLOW}[DEBUG] train/labels exists: $([ -d "$dataset_dir/train/labels" ] && echo "yes" || echo "no")${NC}"
            echo -e "${YELLOW}[DEBUG] images/train exists: $([ -d "$dataset_dir/images/train" ] && echo "yes" || echo "no")${NC}"
            echo -e "${YELLOW}[DEBUG] labels/train exists: $([ -d "$dataset_dir/labels/train" ] && echo "yes" || echo "no")${NC}"
        fi
        return 1
    fi
    
    # Check if directories contain files
    local train_images_count=$(find "$train_images_dir" -type f 2>/dev/null | wc -l)
    local train_labels_count=$(find "$train_labels_dir" -type f 2>/dev/null | wc -l)
    
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}[DEBUG] Train images: $train_images_count${NC}"
        echo -e "${CYAN}[DEBUG] Train labels: $train_labels_count${NC}"
    fi
    
    if [ "$train_images_count" -eq 0 ] || [ "$train_labels_count" -eq 0 ]; then
        if [ "$VERBOSE" = true ]; then
            echo -e "${YELLOW}[DEBUG] No files found in train directories${NC}"
        fi
        return 1
    fi
    
    if [ "$VERBOSE" = true ]; then
        echo -e "${GREEN}[DEBUG] Dataset validation passed ✓${NC}"
    fi
    
    return 0
}

fish_segmentation_to_yolo() {
    print_section "📊 Converting FiftyOne dataset to YOLO segmentation format..."
    echo "Dataset: $FO_FISH_SEGMENTATION_DATASET"
    echo "Output: $YOLO_FISH_SEGMENTATION_OUTPUT_DIR"
    echo "Polyline field: $POLYLINE_FIELD"
    echo ""
    
    python ../module/segmentation_mask/fiftyone_to_yolo_segmentation.py \
        --dataset "$FO_FISH_SEGMENTATION_DATASET" \
        --output_dir "$YOLO_FISH_SEGMENTATION_OUTPUT_DIR" \
        --split_train_val \
        --num_classes "$NUM_CLASSES" \
        --polyline_field "$POLYLINE_FIELD" \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset conversion completed${NC}"
    else
        echo -e "${RED}❌ Dataset conversion failed${NC}"
        exit 1
    fi
    echo ""
}

print_training_config() {
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${BOLD}  Segmentation Training Configuration${NC}"
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ -n "$RESUME_CHECKPOINT" ]; then
        echo -e "${YELLOW}${BOLD}⚠  RESUME MODE${NC}"
        echo -e "${YELLOW}  Resuming from: $RESUME_CHECKPOINT${NC}"
        echo ""
    fi
    
    echo -e "${CYAN}Dataset:${NC}"
    echo "  FiftyOne Dataset: $FO_FISH_SEGMENTATION_DATASET"
    echo "  YOLO Output: $YOLO_FISH_SEGMENTATION_OUTPUT_DIR"
    echo "  Data YAML: $DATA"
    echo "  Classes: $NUM_CLASSES"
    echo "  Polyline Field: $POLYLINE_FIELD"
    echo ""
    
    echo -e "${CYAN}Model & Training:${NC}"
    if [ -n "$RESUME_CHECKPOINT" ]; then
        echo "  Checkpoint: $RESUME_CHECKPOINT"
    else
        echo "  Model: $MODEL"
    fi
    echo "  Project: $PROJECT"
    echo "  Run Name: $RUN_NAME"
    echo "  GPU Device: $GPU_DEVICE"
    echo ""
    
    echo -e "${CYAN}Training Parameters:${NC}"
    echo "  Epochs: $EPOCHS"
    echo "  Batch Size: $BATCH (auto)"
    echo "  Image Size: ${IMGSZ}px"
    echo "  Patience: $PATIENCE"
    echo "  Workers: $WORKERS"
    echo ""
    
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}Segmentation Settings:${NC}"
        echo "  Confidence: $CONF"
        echo "  IoU Threshold: $IOU"
        echo "  Max Detections: $MAX_DET"
        echo "  Overlap Mask: $OVERLAP_MASK"
        echo "  Mask Ratio: $MASK_RATIO"
        echo "  Retina Masks: $RETINA_MASKS"
        echo ""
        
        echo -e "${CYAN}Augmentation:${NC}"
        echo "  Level: $AUGMENT_LEVEL"
        echo "  Mosaic: $MOSAIC"
        echo "  MixUp: $MIXUP"
        echo "  Copy-Paste: $COPY_PASTE"
        echo "  Multi-Scale: $MULTI_SCALE"
        echo ""
        
        echo -e "${CYAN}Optimization:${NC}"
        echo "  Optimizer: $OPTIMIZER"
        echo "  Learning Rate: $LR0"
        echo "  AMP: $AMP"
        echo "  Single Class: $SINGLE_CLS"
        echo ""
    fi
    
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

train_segmentation() {
    print_section "🚀 Training segmentation model..."
    
    # Save configuration before training
    local config_file="$PROJECT/${RUN_NAME}_config.txt"
    mkdir -p "$PROJECT"
    save_config "$config_file"
    echo ""
    
    # Build command
    CMD="python ../train_scripts/segmentation/train_yolo.py"
    
    # Model or Resume
    if [ -n "$RESUME_CHECKPOINT" ]; then
        CMD="$CMD --resume \"$RESUME_CHECKPOINT\""
    else
        CMD="$CMD --model \"$MODEL\""
    fi
    
    CMD="$CMD --data \"$DATA\""
    CMD="$CMD --project \"$PROJECT\""
    CMD="$CMD --run_name \"$RUN_NAME\""
    
    # Training parameters
    CMD="$CMD --epochs $EPOCHS"
    CMD="$CMD --batch $BATCH"
    CMD="$CMD --imgsz $IMGSZ"
    CMD="$CMD --patience $PATIENCE"
    CMD="$CMD --device $GPU_DEVICE"
    CMD="$CMD --workers $WORKERS"
    
    # Segmentation settings
    CMD="$CMD --conf $CONF"
    CMD="$CMD --iou $IOU"
    CMD="$CMD --max_det $MAX_DET"
    CMD="$CMD --mask_ratio $MASK_RATIO"
    
    # Augmentation
    CMD="$CMD --augment_level $AUGMENT_LEVEL"
    CMD="$CMD --mosaic $MOSAIC"
    CMD="$CMD --mixup $MIXUP"
    CMD="$CMD --copy_paste $COPY_PASTE"
    
    # Optimization
    CMD="$CMD --optimizer $OPTIMIZER"
    CMD="$CMD --lr0 $LR0"
    
    # Flags
    if [ "$AMP" = true ]; then
        CMD="$CMD --amp"
    fi
    
    if [ "$MULTI_SCALE" = true ]; then
        CMD="$CMD --multi_scale"
    fi
    
    if [ "$SINGLE_CLS" = true ]; then
        CMD="$CMD --single_cls"
    fi
    
    if [ "$OVERLAP_MASK" = true ]; then
        CMD="$CMD --overlap_mask"
    fi
    
    if [ "$RETINA_MASKS" = true ]; then
        CMD="$CMD --retina_masks"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        CMD="$CMD --dry_run"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    CMD="$CMD --pretrained --exist_ok"
    
    # Execute
    if [ "$QUIET" = false ]; then
        echo "Executing training command..."
        if [ "$VERBOSE" = true ]; then
            echo -e "${CYAN}Command:${NC} $CMD"
        fi
        echo ""
    fi
    
    eval $CMD
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}${BOLD}"
        echo "================================================================================"
        echo "  ✅ TRAINING COMPLETED SUCCESSFULLY!"
        echo "================================================================================"
        echo -e "${NC}"
        echo ""
        echo -e "${CYAN}Results saved to:${NC} $PROJECT/$RUN_NAME"
        echo ""
        echo -e "${CYAN}Best weights:${NC} $PROJECT/$RUN_NAME/weights/best.pt"
        echo -e "${CYAN}Last weights:${NC} $PROJECT/$RUN_NAME/weights/last.pt"
        echo -e "${CYAN}Configuration:${NC} $config_file"
        echo ""
        echo -e "${YELLOW}Next steps:${NC}"
        echo "  1. Validate: python -m ultralytics val model=$PROJECT/$RUN_NAME/weights/best.pt data=$DATA task=segment"
        echo "  2. Predict:  python -m ultralytics predict model=$PROJECT/$RUN_NAME/weights/best.pt source=<image> task=segment"
        echo "  3. Export:   python -m ultralytics export model=$PROJECT/$RUN_NAME/weights/best.pt format=onnx"
        echo "  4. Resume:   ./segmentation_yolo.sh -c $PROJECT/$RUN_NAME/weights/last.pt -r ${RUN_NAME}_resumed"
        echo ""
    else
        echo ""
        echo -e "${RED}${BOLD}"
        echo "================================================================================"
        echo "  ❌ TRAINING FAILED"
        echo "================================================================================"
        echo -e "${NC}"
        echo ""
        echo "Check the error messages above for details."
        echo ""
        exit $EXIT_CODE
    fi
}

# ============== MAIN EXECUTION ==============

print_header

# Validate inputs
echo ""
print_section "🔍 Validating configuration..."
echo ""

# Validate checkpoint if provided
if [ -n "$RESUME_CHECKPOINT" ]; then
    if ! validate_checkpoint "$RESUME_CHECKPOINT"; then
        exit 1
    fi
else
    # Validate model only if not resuming
    if ! validate_model "$MODEL"; then
        exit 1
    fi
fi

# Check GPU
echo ""
check_gpu
echo ""

# Print configuration
print_training_config

# Confirmation (skip in dry run mode)
if [ "$DRY_RUN" = false ] && [ "$QUIET" = false ]; then
    echo -e "${YELLOW}Continue with this configuration? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# Execute pipeline
echo -e "${CYAN}${BOLD}Starting pipeline...${NC}"
echo ""

# Check if dataset already exists
print_section "🔍 Checking for existing dataset..."
echo ""

if check_dataset_exists "$YOLO_FISH_SEGMENTATION_OUTPUT_DIR"; then
    echo -e "${GREEN}✓ YOLO dataset already exists at: $YOLO_FISH_SEGMENTATION_OUTPUT_DIR${NC}"
    echo -e "${YELLOW}⏭  Skipping dataset conversion${NC}"
    echo ""
else
    if [ "$VERBOSE" = true ]; then
        echo -e "${YELLOW}Dataset not found or incomplete, will create new dataset${NC}"
        echo ""
    fi
    fish_segmentation_to_yolo
fi

train_segmentation

echo ""
echo -e "${GREEN}${BOLD}🎉 Segmentation pipeline finished!${NC}"
echo ""
