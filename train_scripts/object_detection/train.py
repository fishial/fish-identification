import os
import sys
import logging
import argparse
import torch
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from ultralytics import YOLO, settings
from datetime import datetime

# ============== CONFIGURATION ==============
settings.update({"wandb": False})
os.environ['WANDB_MODE'] = 'disabled'
DEFAULT_MODEL = "yolo26n.pt"

# Optimal hyperparameters for high-precision fish detection with multiple objects
OPTIMIZED_HYPERPARAMS = {
    # Optimizer settings
    'optimizer': 'AdamW',  # AdamW generally performs better than SGD
    'lr0': 0.001,  # Initial learning rate
    'lrf': 0.01,  # Final learning rate (lr0 * lrf)
    'momentum': 0.937,  # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # Optimizer weight decay
    'warmup_epochs': 3.0,  # Warmup epochs
    'warmup_momentum': 0.8,  # Warmup initial momentum
    'warmup_bias_lr': 0.1,  # Warmup initial bias lr
    
    # Loss weights (tuned for single-class detection)
    'box': 7.5,  # Box loss gain (higher for better localization)
    'cls': 0.5,  # Class loss gain (lower for single class)
    'dfl': 1.5,  # DFL loss gain
    
    # Data augmentation (optimized for underwater/fish detection)
    'hsv_h': 0.015,  # HSV-Hue augmentation (range 0-1)
    'hsv_s': 0.7,  # HSV-Saturation augmentation (range 0-1)
    'hsv_v': 0.4,  # HSV-Value augmentation (range 0-1)
    'degrees': 10.0,  # Image rotation (+/- deg)
    'translate': 0.1,  # Image translation (+/- fraction)
    'scale': 0.9,  # Image scale (+/- gain)
    'shear': 2.0,  # Image shear (+/- deg)
    'perspective': 0.0001,  # Image perspective (+/- fraction)
    'flipud': 0.0,  # Image flip up-down (probability)
    'fliplr': 0.5,  # Image flip left-right (probability)
    'mosaic': 1.0,  # Mosaic augmentation (probability)
    'mixup': 0.15,  # MixUp augmentation (probability)
    'copy_paste': 0.3,  # Copy-paste augmentation (probability)
    'auto_augment': 'randaugment',  # Auto augmentation policy
    'erasing': 0.4,  # Random erasing probability
    
    # Advanced augmentation
    'bgr': 0.0,  # BGR channel shuffle (no shuffle for consistent color)
    
    # Training settings
    'amp': True,  # Automatic Mixed Precision training
    'close_mosaic': 10,  # Disable mosaic in last N epochs
    
    # Detection-specific (CRITICAL for detecting many objects)
    'iou': 0.7,  # IoU threshold for training
    'multi_scale': False,  # Multi-scale training (disabled by default to avoid errors)
    
    # Validation settings
    'save_period': -1,  # Save checkpoint every x epochs (disabled with -1)
    'save_json': True,  # Save results to JSON
    'plots': True,  # Save plots during training
    'cache': 'disk',  # Cache images on disk (use 'disk' to avoid OOM, True for RAM, False to disable)
}

# Inference/Validation settings for detecting multiple fish
INFERENCE_PARAMS = {
    'conf': 0.15,  # Lower confidence threshold to detect more objects
    'iou': 0.45,  # NMS IoU threshold (lower = less aggressive NMS)
    'max_det': 500,  # Maximum detections per image (critical for many fish)
    'agnostic_nms': False,  # Class-agnostic NMS
    'retina_masks': True,  # High-resolution masks
}

# Model-specific recommendations
MODEL_RECOMMENDATIONS = {
    'yolo11n.pt': {'imgsz': 640, 'batch': 32},
    'yolo11s.pt': {'imgsz': 640, 'batch': 24},
    'yolo11m.pt': {'imgsz': 800, 'batch': 16},
    'yolo11l.pt': {'imgsz': 960, 'batch': 8},
    'yolo11x.pt': {'imgsz': 1024, 'batch': 4},
    'yolo26n.pt': {'imgsz': 640, 'batch': 32},
    'yolo26s.pt': {'imgsz': 640, 'batch': 24},
    'yolo26m.pt': {'imgsz': 800, 'batch': 16},
}

# ============== ARGUMENT PARSER ==============
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training a YOLO model with advanced options.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Advanced YOLO training script optimized for high-precision fish detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with optimized defaults
  python train.py --data fish.yaml --project ./runs --run_name fish_detection
  
  # High-precision training with large images
  python train.py --data fish.yaml --project ./runs --run_name fish_hp --imgsz 1280 --epochs 300
  
  # Fast training for testing
  python train.py --data fish.yaml --project ./runs --run_name test --epochs 50 --device 0 --workers 8
  
  # Resume from checkpoint
  python train.py --data fish.yaml --project ./runs --run_name resume --resume ./runs/fish_hp/weights/last.pt
        """
    )
    
    # ===== Model Configuration =====
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to YOLO model or pretrained weights (e.g., yolo11n.pt, yolo26m.pt)"
    )
    model_group.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights (default: True)"
    )
    model_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint path"
    )
    
    # ===== Dataset Configuration =====
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )
    data_group.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0, default: 1.0 for full dataset)"
    )
    data_group.add_argument(
        "--cache",
        type=str,
        choices=['ram', 'disk', 'none'],
        default='ram',
        help="Cache images for faster training (ram/disk/none, default: ram)"
    )
    
    # ===== Training Parameters =====
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300 for high precision)"
    )
    train_group.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto-batch, default: -1)"
    )
    train_group.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size (default: 1280 for high precision, use 640/800/960/1280)"
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stopping patience in epochs (0 to disable, default: 100)"
    )
    train_group.add_argument(
        "--device",
        type=str,
        default='',
        help="CUDA device(s) to use (e.g., '0' or '0,1' or 'cpu', default: auto)"
    )
    train_group.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading (default: 8)"
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    
    # ===== Optimization Parameters =====
    opt_group = parser.add_argument_group('Optimization Parameters')
    opt_group.add_argument(
        "--optimizer",
        type=str,
        choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
        default='AdamW',
        help="Optimizer to use (default: AdamW for best convergence)"
    )
    opt_group.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)"
    )
    opt_group.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate factor (default: 0.01)"
    )
    opt_group.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum / Adam beta1 (default: 0.937)"
    )
    opt_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Optimizer weight decay (default: 0.0005)"
    )
    opt_group.add_argument(
        "--warmup_epochs",
        type=float,
        default=3.0,
        help="Warmup epochs (default: 3.0)"
    )
    opt_group.add_argument(
        "--cos_lr",
        action="store_true",
        default=True,
        help="Use cosine learning rate scheduler (default: True)"
    )
    opt_group.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=True,
        help="Use Automatic Mixed Precision training (default: True)"
    )
    opt_group.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable Automatic Mixed Precision training"
    )
    
    # ===== Detection-Specific Parameters =====
    det_group = parser.add_argument_group('Detection Parameters (for multiple objects)')
    det_group.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detection (lower = more detections, default: 0.15)"
    )
    det_group.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (lower = less aggressive NMS, default: 0.45)"
    )
    det_group.add_argument(
        "--max_det",
        type=int,
        default=500,
        help="Maximum detections per image (default: 500 for many fish)"
    )
    
    # ===== Augmentation Parameters =====
    aug_group = parser.add_argument_group('Data Augmentation')
    aug_group.add_argument(
        "--augment_level",
        type=str,
        choices=['none', 'light', 'medium', 'heavy', 'custom'],
        default='heavy',
        help="Augmentation preset (default: heavy for max accuracy)"
    )
    aug_group.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (default: 1.0)"
    )
    aug_group.add_argument(
        "--mixup",
        type=float,
        default=0.15,
        help="MixUp augmentation probability (default: 0.15)"
    )
    aug_group.add_argument(
        "--copy_paste",
        type=float,
        default=0.3,
        help="Copy-paste augmentation probability (default: 0.3)"
    )
    aug_group.add_argument(
        "--degrees",
        type=float,
        default=10.0,
        help="Image rotation degrees (default: 10.0)"
    )
    aug_group.add_argument(
        "--scale",
        type=float,
        default=0.9,
        help="Image scale factor (default: 0.9)"
    )
    aug_group.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Horizontal flip probability (default: 0.5)"
    )
    
    # ===== Output Configuration =====
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project directory for training outputs"
    )
    output_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (default: auto-generated with timestamp)"
    )
    output_group.add_argument(
        "--save_period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (-1 to disable, default: 10)"
    )
    output_group.add_argument(
        "--exist_ok",
        action="store_true",
        help="Allow overwriting existing run directory"
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    
    # ===== Advanced Options =====
    adv_group = parser.add_argument_group('Advanced Options')
    adv_group.add_argument(
        "--multi_scale",
        action="store_true",
        default=False,
        help="Multi-scale training (default: False to avoid interpolation errors, use --multi_scale to enable)"
    )
    adv_group.add_argument(
        "--rect",
        action="store_true",
        help="Rectangular training (faster but less accurate)"
    )
    adv_group.add_argument(
        "--overlap_mask",
        action="store_true",
        default=True,
        help="Masks should overlap during training (default: True)"
    )
    adv_group.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon (default: 0.0)"
    )
    adv_group.add_argument(
        "--single_cls",
        action="store_true",
        help="Train as single-class dataset (recommended for fish detection)"
    )
    adv_group.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
        help="Disable mosaic in last N epochs (default: 10)"
    )
    
    # ===== Utilities =====
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without training"
    )
    util_group.add_argument(
        "--profile",
        action="store_true",
        help="Profile ONNX model speed"
    )
    util_group.add_argument(
        "--auto_optimize",
        action="store_true",
        help="Automatically optimize hyperparameters based on model size"
    )

    args = parser.parse_args()

    # Auto-generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(args.model).stem
        args.run_name = f"fish_detection_{model_name}_{timestamp}"

    # Validate fraction value
    if not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be between 0 (exclusive) and 1.0 (inclusive)")
    
    # Validate confidence threshold
    if not (0.0 < args.conf < 1.0):
        parser.error("--conf must be between 0 and 1")
    
    # Validate IoU threshold
    if not (0.0 < args.iou < 1.0):
        parser.error("--iou must be between 0 and 1")
        
    return args

def validate_dataset_file(data_path: Path) -> Dict[str, Any]:
    """
    Validate that the dataset file exists and load its contents.

    Args:
        data_path (Path): Path to the dataset file.

    Returns:
        Dict[str, Any]: Loaded dataset configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not have a valid YAML extension or is malformed.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML file not found at {data_path}")
    
    if data_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Dataset file should have a .yaml or .yml extension, got {data_path.suffix}")
    
    # Load and validate YAML content
    try:
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        if not isinstance(data_config, dict):
            raise ValueError("Dataset YAML must contain a dictionary")
        
        # Check for required fields
        required_fields = ['path', 'train', 'val', 'names']
        missing_fields = [field for field in required_fields if field not in data_config]
        if missing_fields:
            logging.warning(f"Dataset YAML missing recommended fields: {missing_fields}")
        
        # Log dataset info
        if 'names' in data_config:
            num_classes = len(data_config['names']) if isinstance(data_config['names'], (list, dict)) else 0
            logging.info(f"Dataset contains {num_classes} class(es)")
            if num_classes == 1:
                logging.info("✓ Single-class detection - perfect for fish detection!")
        
        return data_config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse dataset YAML: {e}")


def get_optimal_hyperparams(args: argparse.Namespace, data_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get optimized hyperparameters based on arguments and dataset.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        data_config (Dict[str, Any]): Dataset configuration.

    Returns:
        Dict[str, Any]: Optimized hyperparameters.
    """
    hyperparams = OPTIMIZED_HYPERPARAMS.copy()
    
    # Override with command-line arguments
    if hasattr(args, 'optimizer'):
        hyperparams['optimizer'] = args.optimizer
    if hasattr(args, 'lr0'):
        hyperparams['lr0'] = args.lr0
    if hasattr(args, 'lrf'):
        hyperparams['lrf'] = args.lrf
    if hasattr(args, 'momentum'):
        hyperparams['momentum'] = args.momentum
    if hasattr(args, 'weight_decay'):
        hyperparams['weight_decay'] = args.weight_decay
    if hasattr(args, 'warmup_epochs'):
        hyperparams['warmup_epochs'] = args.warmup_epochs
    if hasattr(args, 'amp'):
        hyperparams['amp'] = args.amp
    if hasattr(args, 'mosaic'):
        hyperparams['mosaic'] = args.mosaic
    if hasattr(args, 'mixup'):
        hyperparams['mixup'] = args.mixup
    if hasattr(args, 'copy_paste'):
        hyperparams['copy_paste'] = args.copy_paste
    if hasattr(args, 'degrees'):
        hyperparams['degrees'] = args.degrees
    if hasattr(args, 'scale'):
        hyperparams['scale'] = args.scale
    if hasattr(args, 'fliplr'):
        hyperparams['fliplr'] = args.fliplr
    if hasattr(args, 'multi_scale'):
        hyperparams['multi_scale'] = args.multi_scale
    if hasattr(args, 'close_mosaic'):
        hyperparams['close_mosaic'] = args.close_mosaic
    
    # Adjust based on augmentation level
    if args.augment_level == 'none':
        hyperparams.update({
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 0.0,
            'scale': 0.0,
            'fliplr': 0.0,
        })
    elif args.augment_level == 'light':
        hyperparams.update({
            'mosaic': 0.5,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 5.0,
            'scale': 0.5,
            'fliplr': 0.5,
        })
    elif args.augment_level == 'medium':
        hyperparams.update({
            'mosaic': 0.8,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'degrees': 10.0,
            'scale': 0.7,
            'fliplr': 0.5,
        })
    # 'heavy' keeps defaults, 'custom' uses command-line overrides
    
    # Single-class optimization
    num_classes = len(data_config.get('names', [])) if isinstance(data_config.get('names'), (list, dict)) else 1
    if num_classes == 1 or args.single_cls:
        hyperparams['cls'] = 0.3  # Lower class loss for single class
        logging.info("✓ Applying single-class optimization")
    
    return hyperparams


def get_model_recommendations(model_path: str) -> Dict[str, Any]:
    """
    Get recommended settings for specific model architecture.

    Args:
        model_path (str): Path to the model file.

    Returns:
        Dict[str, Any]: Recommended settings.
    """
    model_name = Path(model_path).name
    
    # Check if we have specific recommendations
    if model_name in MODEL_RECOMMENDATIONS:
        return MODEL_RECOMMENDATIONS[model_name].copy()
    
    # Default recommendations
    return {'imgsz': 640, 'batch': 16}


def log_training_config(args: argparse.Namespace, hyperparams: Dict[str, Any]) -> None:
    """
    Log the complete training configuration.

    Args:
        args (argparse.Namespace): Training arguments.
        hyperparams (Dict[str, Any]): Hyperparameters.
    """
    logging.info("=" * 80)
    logging.info("🚀 ADVANCED FISH DETECTION TRAINING")
    logging.info("=" * 80)
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.data}")
    logging.info(f"Output: {Path(args.project) / args.run_name}")
    logging.info("-" * 80)
    logging.info("Training Configuration:")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Batch Size: {args.batch if args.batch > 0 else 'auto'}")
    logging.info(f"  Image Size: {args.imgsz}")
    logging.info(f"  Device: {args.device if args.device else 'auto'}")
    logging.info(f"  Workers: {args.workers}")
    logging.info(f"  Patience: {args.patience}")
    logging.info("-" * 80)
    logging.info("Optimization:")
    logging.info(f"  Optimizer: {hyperparams.get('optimizer', 'AdamW')}")
    logging.info(f"  Learning Rate: {hyperparams.get('lr0', 0.001)} → {hyperparams.get('lr0', 0.001) * hyperparams.get('lrf', 0.01)}")
    logging.info(f"  Weight Decay: {hyperparams.get('weight_decay', 0.0005)}")
    logging.info(f"  Warmup Epochs: {hyperparams.get('warmup_epochs', 3.0)}")
    logging.info(f"  AMP Training: {hyperparams.get('amp', True)}")
    logging.info(f"  Cosine LR: {args.cos_lr}")
    logging.info("-" * 80)
    logging.info("Detection Settings (for multiple objects):")
    logging.info(f"  Confidence Threshold: {args.conf}")
    logging.info(f"  NMS IoU Threshold: {args.iou}")
    logging.info(f"  Max Detections: {args.max_det}")
    logging.info("-" * 80)
    logging.info("Augmentation:")
    logging.info(f"  Level: {args.augment_level}")
    logging.info(f"  Mosaic: {hyperparams.get('mosaic', 1.0)}")
    logging.info(f"  MixUp: {hyperparams.get('mixup', 0.15)}")
    logging.info(f"  Copy-Paste: {hyperparams.get('copy_paste', 0.3)}")
    logging.info(f"  Multi-Scale: {hyperparams.get('multi_scale', True)}")
    logging.info("=" * 80)
    
    # GPU info
    if torch.cuda.is_available():
        logging.info(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        logging.info(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logging.warning("⚠ No GPU detected - training will be slow!")
    logging.info("=" * 80)

# ============== TRAINING FUNCTION ==============
def train_yolo_model(args: argparse.Namespace) -> Any:
    """
    Train a YOLO model with advanced optimization for fish detection.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        Any: Training results from the YOLO model.
    """
    # Validate and load dataset
    data_path = Path(args.data)
    data_config = validate_dataset_file(data_path)

    # Get optimized hyperparameters
    hyperparams = get_optimal_hyperparams(args, data_config)
    
    # Get model recommendations
    if args.auto_optimize:
        recommendations = get_model_recommendations(args.model)
        if args.imgsz == 640:  # Only auto-adjust if user didn't specify custom size
            args.imgsz = recommendations.get('imgsz', 640)
            logging.info(f"✓ Auto-optimized image size to {args.imgsz}")
        if args.batch == -1:
            args.batch = recommendations.get('batch', -1)
    
    # Create the output directory
    output_dir = Path(args.project) / args.run_name
    if not args.exist_ok and output_dir.exists():
        logging.warning(f"Output directory already exists: {output_dir}")
        logging.warning("Use --exist_ok to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    log_training_config(args, hyperparams)

    # Load YOLO model
    logging.info(f"Loading YOLO model: {args.model}")
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(args.model)

    # Prepare training arguments
    train_args = {
        # Basic settings
        'data': str(data_path),
        'project': str(Path(args.project)),
        'name': args.run_name,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device if args.device else None,
        'workers': args.workers,
        'pretrained': args.pretrained,
        'seed': args.seed,
        
        # Dataset settings
        'fraction': args.fraction,
        'cache': args.cache if args.cache != 'none' else False,
        
        # Training control
        'patience': args.patience,
        'save_period': args.save_period,
        'exist_ok': args.exist_ok,
        'verbose': args.verbose,
        'resume': bool(args.resume),
        
        # Detection-specific (critical for many objects)
        'conf': args.conf,
        'iou': args.iou,
        'max_det': args.max_det,
        
        # Advanced settings
        'rect': args.rect,
        'overlap_mask': args.overlap_mask,
        'label_smoothing': args.label_smoothing,
        'single_cls': args.single_cls,
        'cos_lr': args.cos_lr,
        'profile': args.profile,
        
        # Plots and logging
        'plots': True,
        'save_json': True,
    }
    
    # Merge hyperparameters
    train_args.update(hyperparams)
    
    # Remove None values
    train_args = {k: v for k, v in train_args.items() if v is not None}

    # Start training
    logging.info(f"🚀 Starting training run: {args.run_name}")
    logging.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = model.train(**train_args)
        
        logging.info("=" * 80)
        logging.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 80)
        logging.info(f"📁 Results saved in: {output_dir}")
        logging.info(f"🏆 Best weights: {output_dir / 'weights' / 'best.pt'}")
        logging.info(f"💾 Last weights: {output_dir / 'weights' / 'last.pt'}")
        
        # Log best metrics if available
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            logging.info("-" * 80)
            logging.info("Best Metrics:")
            if 'metrics/mAP50(B)' in metrics:
                logging.info(f"  mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                logging.info(f"  mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/precision(B)' in metrics:
                logging.info(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                logging.info(f"  Recall: {metrics['metrics/recall(B)']:.4f}")
        
        logging.info("=" * 80)
        
        # Export model info
        export_training_summary(args, output_dir, hyperparams)
        
        return results
        
    except KeyboardInterrupt:
        logging.warning("\n⚠ Training interrupted by user")
        logging.info(f"Partial results saved in: {output_dir}")
        raise
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise


def export_training_summary(args: argparse.Namespace, output_dir: Path, hyperparams: Dict[str, Any]) -> None:
    """
    Export training configuration summary to file.

    Args:
        args (argparse.Namespace): Training arguments.
        output_dir (Path): Output directory.
        hyperparams (Dict[str, Any]): Hyperparameters used.
    """
    summary_file = output_dir / "training_config.yaml"
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'dataset': args.data,
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch,
            'image_size': args.imgsz,
            'device': args.device,
            'workers': args.workers,
            'patience': args.patience,
            'fraction': args.fraction,
        },
        'detection_params': {
            'conf_threshold': args.conf,
            'iou_threshold': args.iou,
            'max_detections': args.max_det,
        },
        'hyperparameters': hyperparams,
        'augmentation_level': args.augment_level,
    }
    
    try:
        with open(summary_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
        logging.info(f"📝 Training config saved to: {summary_file}")
    except Exception as e:
        logging.warning(f"Failed to save training summary: {e}")

# ============== MAIN EXECUTION ==============
def main():
    """Main execution function."""
    # Configure logging with timestamps and levels
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    if args.seed:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Dry-run mode: validate configuration and exit without training
    if args.dry_run:
        logging.info("=" * 80)
        logging.info("🔍 DRY RUN MODE - Configuration Validation")
        logging.info("=" * 80)
        
        # Validate dataset
        data_path = Path(args.data)
        try:
            data_config = validate_dataset_file(data_path)
            logging.info("✓ Dataset file valid")
            
            # Get hyperparameters
            hyperparams = get_optimal_hyperparams(args, data_config)
            
            # Show configuration
            log_training_config(args, hyperparams)
            
            # Check model file
            if not Path(args.model).exists():
                logging.warning(f"⚠ Model file not found: {args.model}")
                logging.info("Model will be downloaded from Ultralytics on first run")
            else:
                logging.info(f"✓ Model file exists: {args.model}")
            
            # Estimate training time
            approx_time_per_epoch = args.imgsz * args.imgsz * 0.0001  # rough estimate in seconds
            estimated_time = approx_time_per_epoch * args.epochs / 60  # minutes
            logging.info("-" * 80)
            logging.info(f"⏱ Estimated training time: ~{estimated_time:.1f} minutes")
            logging.info("(Actual time depends on hardware, dataset size, and batch size)")
            logging.info("-" * 80)
            
            logging.info("✅ Configuration validation successful!")
            logging.info("Remove --dry_run flag to start actual training")
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
        
        sys.exit(0)
    
    # Normal training mode
    try:
        start_time = datetime.now()
        results = train_yolo_model(args)
        end_time = datetime.now()
        
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logging.info("=" * 80)
        logging.info(f"⏱ Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        logging.info(f"⏰ Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 80)
        logging.info("🎉 Training pipeline completed successfully!")
        logging.info("")
        logging.info("Next steps:")
        logging.info(f"  1. Validate model: python -m ultralytics val model={Path(args.project) / args.run_name / 'weights' / 'best.pt'}")
        logging.info(f"  2. Test inference: python -m ultralytics predict model={Path(args.project) / args.run_name / 'weights' / 'best.pt'} source=<test_image>")
        logging.info(f"  3. Export model: python -m ultralytics export model={Path(args.project) / args.run_name / 'weights' / 'best.pt'} format=onnx")
        logging.info("=" * 80)
        
    except KeyboardInterrupt:
        logging.warning("\n" + "=" * 80)
        logging.warning("⚠ Training interrupted by user (Ctrl+C)")
        logging.warning("Partial results may be available in the output directory")
        logging.warning("=" * 80)
        sys.exit(130)
    except Exception as e:
        logging.error("=" * 80)
        logging.error("❌ TRAINING FAILED")
        logging.error("=" * 80)
        logging.error(f"Error: {e}", exc_info=True)
        logging.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()