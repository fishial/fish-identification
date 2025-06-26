# -*- coding: utf-8 -*-
"""
Training Script for Image Embedding Models with PyTorch Lightning.

This script orchestrates the training of a Vision Transformer (ViT) based model
for metric learning using the ArcFace loss. It is designed to be highly
configurable and runnable from the command line.

Key Features:
-   Configurable model architecture (embedding size, backbone).
-   Uses PyTorch Lightning for clean and robust training loops.
-   Integrates with FiftyOne for dataset management.
-   Implements a custom `LightningDataModule` for encapsulated data handling.
-   Uses a balanced batch sampler (`MPerClassSampler`) for effective metric learning.
-   Supports resuming training from checkpoints.
-   Comprehensive logging with TensorBoard.
-   All critical parameters are exposed as command-line arguments.

Example Usage:
----------------
# Start a new training run with default parameters
python train_script.py \
    --dataset_name "my_fiftyone_dataset" \
    --output_dir "/path/to/experiments/run_01"

# Resume training from a checkpoint, using a different learning rate
python train_script.py \
    --dataset_name "my_fiftyone_dataset" \
    --output_dir "/path/to/experiments/run_01" \
    --resume_from_checkpoint "/path/to/experiments/run_01/checkpoints/last.ckpt" \
    --learning_rate 5e-5

# Run in debug mode on a small subset of data for 5 epochs
python train_script.py \
    --dataset_name "my_fiftyone_dataset" \
    --output_dir "/path/to/experiments/debug_run" \
    --limit_train_batches 10 \
    --limit_val_batches 10 \
    --max_epochs 5
"""

# --- Standard Library Imports ---
import argparse
import logging
import math
import os
import random
import sys
from typing import Optional, Dict, List

# --- Third-Party Imports ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# PyTorch Lightning
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

# Metric Learning
from pytorch_metric_learning import samplers

# Data and Augmentations
import fiftyone as fo
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import custom modules
# Modify sys.path to include the root directory containing 'fish-identification'
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
    print("SETUP: sys.path updated")
    
from module.classification_package.src.lightning_trainer import ImageEmbeddingTrainerConvnext, ImageEmbeddingTrainerViT
from module.classification_package.src.datamodule import ImageEmbeddingDataModule

# Setup professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =================================================================================
# SCRIPT EXECUTION
# =================================================================================

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an Image Embedding Model.")

    # --- Data and Paths ---
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the FiftyOne dataset to use.")
    parser.add_argument("--visualize_attention_map", type=bool, default=True, help="the visualizations of random attention maps will be saved during validation at each epoch.")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")

    # --- Model Hyperparameters ---
    parser.add_argument("--backbone_model_name", type=str, default="convnext_tiny", help="Name of the timm backbone model.")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Dimensionality of the final embedding.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for validation loader.")
    parser.add_argument("--classes_per_batch", type=int, default=8, help="Number of classes per batch for MPerClassSampler.")
    parser.add_argument("--samples_per_class", type=int, default=8, help="Number of samples per class for MPerClassSampler.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--lr_eta_min", type=float, default=1e-7, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--attention_loss_lambda", type=float, default=0.15, help="Weight for the attention guidance loss.")
    
    # --- Resuming and Debugging ---
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Fraction of training data to use (e.g., 0.1 for 10%%).")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Fraction of validation data to use.")
    
    return parser.parse_args()

def main(args):
    """Main function to set up and run the training process."""
    pl.seed_everything(42, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Setup DataModule
    datamodule = ImageEmbeddingDataModule(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        image_size=224, # Typically fixed by the model
        num_workers=args.num_workers
    )
    # This needs to be called to create the label mappings and get num_classes
    datamodule.setup()
    
    # 2. Setup Model
    
    trainer = ImageEmbeddingTrainerConvnext if args.backbone_model_name == 'convnext_tiny' else ImageEmbeddingTrainerViT
    model = trainer(
        num_classes=datamodule.num_classes,
        embedding_dim=args.embedding_dim,
        backbone_model_name=args.backbone_model_name,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_eta_min=args.lr_eta_min,
        attention_loss_lambda=args.attention_loss_lambda,
        output_dir = args.output_dir,
        visualize_attention_map = args.visualize_attention_map
    )
    
    # 3. Setup Callbacks and Logger
    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy_epoch",
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="model-{epoch:02d}-{val/accuracy:.4f}",
        save_top_k=3,
        mode="max",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tensorboard_logger = TensorBoardLogger(save_dir=args.output_dir, name="logs")

    # 4. Setup Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.max_epochs,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        deterministic=True
    )

    # 5. Start Training
    logger.info("Starting model training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )
    logger.info("Training finished successfully.")


if __name__ == "__main__":
    cli_args = get_args()
    main(cli_args)
