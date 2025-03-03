import os
import logging
import argparse
from ultralytics import YOLO


# ============== CONFIGURATION ==============
DEFAULT_MODEL = "yolov10s.yaml"
DEFAULT_DATASET = "/home/andrew/Datasets/data.yaml"
DEFAULT_PROJECT = "UltraSegmTrainFISHIAL_OBJECT_DETECTION"
DEFAULT_RUN_NAME = "yolov10s_640_False"

# ============== ARGUMENT PARSER ==============
def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model with custom parameters.")
    
    # Required Arguments
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to YOLO model config file (e.g., yolov10s.yaml)")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET, help="Path to dataset YAML file")
    
    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (square)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (1.0 for full dataset)")
    parser.add_argument("--patience", type=int, default=150, help="Early stopping patience (0 to disable)")
    parser.add_argument("--pretrained", action="store_true", help="Use a pretrained model")

    # Output Configuration
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="Project directory for YOLO training outputs")
    parser.add_argument("--run_name", type=str, default=DEFAULT_RUN_NAME, help="Name for the training run")

    return parser.parse_args()


# ============== TRAINING FUNCTION ==============
def train_yolo_model(args):
    """Train a YOLO model with parsed arguments."""
    
    # Ensure dataset exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset YAML file not found at {args.data}")

    # Initialize logging
    logging.info(f"Loading YOLO model: {args.model}")
    
    # Load YOLO model
    model = YOLO(args.model)

    logging.info(f"Starting training: {args.run_name} (Project: {args.project})")

    # Train the model
    results = model.train(
        data=args.data,
        project=args.project,
        name=args.run_name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        fraction=args.fraction,
        patience=args.patience,
        pretrained=args.pretrained
    )

    logging.info(f"Training completed! Results saved in: {args.project}/{args.run_name}")
    return results


# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Set logging level
    args = parse_args()  # Parse arguments
    train_yolo_model(args)  # Train YOLO model