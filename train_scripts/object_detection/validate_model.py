#!/usr/bin/env python3
"""
🔍 Advanced Model Validation Script

Валидация обученной модели с оптимизированными параметрами
для детекции множества объектов.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate YOLO fish detection model with optimized settings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights (e.g., ./runs/fish_detection/best.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size for validation (default: 1280)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold (default: 0.15 for multiple detections)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45 for multiple detections)"
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=500,
        help="Maximum detections per image (default: 500)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='',
        help="Device to use (e.g., '0', 'cpu', default: auto)"
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--save_hybrid",
        action="store_true",
        help="Save hybrid version of labels (labels + additional predictions)"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Save validation plots (default: True)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="./validation_results",
        help="Project directory for outputs (default: ./validation_results)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fish_val",
        help="Validation run name (default: fish_val)"
    )
    
    return parser.parse_args()


def validate_model(args):
    """
    Validate the model with optimized settings for fish detection.
    
    Args:
        args: Parsed command line arguments
    """
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    
    logging.info("=" * 80)
    logging.info("🔍 ADVANCED FISH DETECTION VALIDATION")
    logging.info("=" * 80)
    logging.info(f"Model: {model_path}")
    logging.info(f"Dataset: {data_path}")
    logging.info(f"Image Size: {args.imgsz}")
    logging.info(f"Batch Size: {args.batch}")
    logging.info("-" * 80)
    logging.info("Detection Settings:")
    logging.info(f"  Confidence Threshold: {args.conf}")
    logging.info(f"  NMS IoU Threshold: {args.iou}")
    logging.info(f"  Max Detections: {args.max_det}")
    logging.info("=" * 80)
    
    # Load model
    logging.info("Loading model...")
    model = YOLO(str(model_path))
    
    # Run validation
    logging.info("Starting validation...")
    results = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device if args.device else None,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        plots=args.plots,
        verbose=args.verbose,
        project=args.project,
        name=args.name,
    )
    
    # Print results
    logging.info("=" * 80)
    logging.info("✅ VALIDATION COMPLETED")
    logging.info("=" * 80)
    
    if hasattr(results, 'box'):
        metrics = results.box
        logging.info("Metrics:")
        logging.info(f"  mAP@0.5: {metrics.map50:.4f}")
        logging.info(f"  mAP@0.5:0.95: {metrics.map:.4f}")
        logging.info(f"  Precision: {metrics.mp:.4f}")
        logging.info(f"  Recall: {metrics.mr:.4f}")
        
        # Additional statistics
        if hasattr(results, 'speed'):
            speed = results.speed
            logging.info("-" * 80)
            logging.info("Speed:")
            logging.info(f"  Preprocess: {speed['preprocess']:.1f}ms")
            logging.info(f"  Inference: {speed['inference']:.1f}ms")
            logging.info(f"  Postprocess: {speed['postprocess']:.1f}ms")
            total_time = speed['preprocess'] + speed['inference'] + speed['postprocess']
            fps = 1000 / total_time if total_time > 0 else 0
            logging.info(f"  Total: {total_time:.1f}ms ({fps:.1f} FPS)")
    
    logging.info("=" * 80)
    logging.info(f"Results saved to: {Path(args.project) / args.name}")
    logging.info("=" * 80)
    
    return results


def main():
    """Main execution function."""
    try:
        args = parse_args()
        validate_model(args)
        
        logging.info("")
        logging.info("🎉 Validation completed successfully!")
        logging.info("")
        logging.info("Next steps:")
        logging.info("  • Check the validation plots in the results directory")
        logging.info("  • If precision is low, increase --conf threshold")
        logging.info("  • If recall is low, decrease --conf threshold")
        logging.info("  • For dense scenes, adjust --iou and --max_det")
        
    except Exception as e:
        logging.error(f"❌ Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
