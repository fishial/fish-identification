import os
import logging
from pathlib import Path
from ultralytics import YOLO, settings

# ============== CONFIGURATION ==============
# Explicitly enable TensorBoard and disable W&B
settings.update({"runs_dir": "/home/fishial/Fishial/Experiments/v10/detection/runs"})
settings.update({"wandb": False})
os.environ['WANDB_MODE'] = 'disabled'

# Paths
PROJECT_ROOT = Path("/home/fishial/Fishial/Experiments/v10/detection")
DATA_CONFIG = PROJECT_ROOT / "segmentation_merged_v0_1_full/data.yaml"
WEIGHTS_PATH = PROJECT_ROOT / "SIMPLE_MEDIUM/weights/best.pt"
RUN_NAME = "simple_training_v2"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def train_model():
    """
    Initializes the YOLO model and starts the training process 
    while logging events to TensorBoard.
    """
    
    # 1. Load Model
    if not WEIGHTS_PATH.exists():
        logging.error(f"Weights file not found at {WEIGHTS_PATH}")
        return

    logging.info(f"🚀 Loading model: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    # 2. Prepare Training Arguments
    train_args = {
        'data': str(DATA_CONFIG),
        'project': str(PROJECT_ROOT),
        'name': RUN_NAME,
        'epochs': 300,
        'batch': 32,
        'imgsz': 640,
        'device': "cuda",
        'workers': 4,
        'seed': 0,
        'cache': "disk",
        'exist_ok': True,

        # --- Optimization ---
        'lr0': 0.02,
        'cos_lr': True,
        'amp': True,

        # --- Logging ---
        'plots': True,
        'save_json': True,
        'tensorboard': True, # Explicitly ensure TensorBoard logging is active
    }

    # 3. Execution
    logging.info(f"📅 Starting training run: {RUN_NAME}")
    logging.info(f"📊 TensorBoard logs will be saved to: {PROJECT_ROOT / RUN_NAME}")
    
    results = model.train(**train_args)

    # 4. Final Summary
    log_final_metrics(results, PROJECT_ROOT / RUN_NAME)

def log_final_metrics(results, output_dir):
    logging.info("=" * 60)
    logging.info("✅ TRAINING COMPLETED")
    
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        metric_map = {
            'metrics/mAP50(B)': 'Box mAP@0.5',
            'metrics/mAP50(M)': 'Mask mAP@0.5',
        }
        for key, label in metric_map.items():
            if key in metrics:
                logging.info(f"  {label:<20}: {metrics[key]:.4f}")
    logging.info("=" * 60)

if __name__ == "__main__":
    train_model()