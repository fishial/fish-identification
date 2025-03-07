import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Any
from ultralytics import YOLO, settings

# ============== CONFIGURATION ==============
settings.update({"wandb": False})
os.environ['WANDB_MODE'] = 'disabled'
DEFAULT_MODEL = "yolov10s.yaml"

# ============== ARGUMENT PARSER ==============
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training a YOLO model.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a YOLO model with custom parameters."
    )
    
    # Required Arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to YOLO model config file (e.g., yolov10s.yaml)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )
    
    # Training Parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square)"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (1.0 for full dataset)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=150,
        help="Early stopping patience (0 to disable)"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use a pretrained model"
    )
    
    # Extra feature: Dry-run mode to test configurations without training
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform a dry run without starting training."
    )

    # Output Configuration
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Local directory for YOLO training outputs (full path)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name for the training run"
    )

    args = parser.parse_args()

    # Validate fraction value
    if not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be between 0 (exclusive) and 1.0 (inclusive)")
        
    return args

def validate_dataset_file(data_path: Path) -> None:
    """
    Validate that the dataset file exists and has a YAML extension.

    Args:
        data_path (Path): Path to the dataset file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not have a valid YAML extension.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML file not found at {data_path}")
    
    if data_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Dataset file should have a .yaml or .yml extension, got {data_path.suffix}")

# ============== TRAINING FUNCTION ==============
def train_yolo_model(args: argparse.Namespace) -> Any:
    """
    Train a YOLO model using the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        Any: Training results from the YOLO model.
    """
    data_path = Path(args.data)
    validate_dataset_file(data_path)

    # Create the output directory if it doesn't exist
    output_dir = Path(args.project) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Local output folder is set to: {output_dir}")

    # Load YOLO model
    logging.info(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    logging.info(f"Starting training run: {args.run_name}")
    results = model.train(
        data=str(data_path),
        project=str(Path(args.project)),
        name=args.run_name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        patience=args.patience,
        pretrained=args.pretrained
    )
    logging.info(f"Training completed! Results saved in: {output_dir}")
    return results

# ============== MAIN EXECUTION ==============
def main():
    # Configure logging with timestamps and levels
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    args = parse_args()
    
    # Dry-run mode: validate configuration and exit without training.
    if args.dry_run:
        logging.info("Dry run mode enabled. Configuration validated successfully:")
        logging.info(f"Arguments: {args}")
        sys.exit(0)
    
    try:
        train_yolo_model(args)
    except Exception as e:
        logging.error("An error occurred during training", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()