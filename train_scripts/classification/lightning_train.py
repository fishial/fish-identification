# -*- coding: utf-8 -*-
"""
Training Script for Image Embedding Models with PyTorch Lightning.

This script orchestrates the training of image embedding models for fish classification
using various backbone architectures and loss functions.

Key Features:
-   Configurable model architecture (ViT, ConvNeXt, EfficientNet, etc.)
-   Multiple loss function options (Combined, CombinedV2 with Focal Loss, etc.)
-   Multiple pooling strategies (Attention, GeM, Hybrid)
-   Configurable data augmentation presets
-   MixUp augmentation support
-   Uses PyTorch Lightning for clean training loops
-   Integrates with FiftyOne for dataset management
-   Comprehensive logging with TensorBoard

Example Usage:
----------------
# Basic training with default settings (backward compatible)
python lightning_train.py \\
    --dataset_name "my_fiftyone_dataset" \\
    --output_dir "/path/to/experiments"

# Training with improved loss (CombinedV2 with Focal Loss)
python lightning_train.py \\
    --dataset_name "my_fiftyone_dataset" \\
    --output_dir "/path/to/experiments" \\
    --loss_type combined_v2 \\
    --metric_loss_type multi_similarity \\
    --focal_weight 0.1 \\
    --augmentation_preset strong

# Training with MixUp and GeM pooling
python lightning_train.py \\
    --dataset_name "my_fiftyone_dataset" \\
    --output_dir "/path/to/experiments" \\
    --use_mixup true \\
    --pooling_type gem \\
    --backbone_model_name convnext_base

# Using EVA-02 backbone with strong augmentations
python lightning_train.py \\
    --dataset_name "my_fiftyone_dataset" \\
    --output_dir "/path/to/experiments" \\
    --backbone_model_name eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \\
    --image_size 448 \\
    --augmentation_preset strong \\
    --loss_type combined_v2
"""

# --- Standard Library Imports ---
import argparse
import logging
import math
import os
import random
import sys
import subprocess
import json
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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

# Metric Learning
from pytorch_metric_learning import samplers

# Data and Augmentations
import fiftyone as fo
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import custom modules
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
    print("SETUP: sys.path updated")
    
from module.classification_package.src.lightning_trainer_fixed import (
    ImageEmbeddingTrainerConvnext, 
    ImageEmbeddingTrainerViT
)
from module.classification_package.src.datamodule import ImageEmbeddingDataModule

# Setup professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetricsToFileCallback(pl.Callback):
    """
    Writes callback_metrics at the end of each validation epoch to:
      - metrics_history.jsonl
      - metrics_history.csv
    inside the current run directory.
    """
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "metrics_history.jsonl")
        self.csv_path = os.path.join(output_dir, "metrics_history.csv")
        self._csv_header_written = False
        self._csv_keys: List[str] = []

    @staticmethod
    def _to_py(v):
        try:
            import torch
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
        except Exception:
            pass
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        try:
            return float(v)
        except Exception:
            return str(v)

    def _write_csv_row(self, row: dict):
        import csv

        if not self._csv_header_written:
            self._csv_keys = list(row.keys())
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_keys)
                w.writeheader()
            self._csv_header_written = True

        out_row = {k: row.get(k, "") for k in self._csv_keys}
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_keys)
            w.writerow(out_row)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        metrics = trainer.callback_metrics
        row = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
        }
        for k, v in metrics.items():
            if k.startswith("hp_metric"):
                continue
            row[k] = self._to_py(v)

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._write_csv_row(row)


# =================================================================================
# ARGUMENT PARSING
# =================================================================================

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an Image Embedding Model with configurable loss and augmentations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (backward compatible)
  python lightning_train.py --dataset_name my_dataset --output_dir ./experiments

  # With improved loss function
  python lightning_train.py --dataset_name my_dataset --output_dir ./experiments \\
      --loss_type combined_v2 --metric_loss_type multi_similarity --focal_weight 0.1

  # With strong augmentations and MixUp
  python lightning_train.py --dataset_name my_dataset --output_dir ./experiments \\
      --augmentation_preset strong --use_mixup true --mixup_alpha 0.4
        """
    )

    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v!r}")

    def parse_exclude_classes(values) -> Optional[List[str]]:
        if values is None:
            return None
        if isinstance(values, str):
            values = [values]
        out: List[str] = []
        for v in values:
            if v is None:
                continue
            for token in str(v).split(","):
                token = token.strip()
                if token:
                    out.append(token)
        return out or None

    def parse_optional_tag(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if value.lower() in {"none", "null", ""}:
            return None
        return value

    # ==========================================================================
    # Data and Paths
    # ==========================================================================
    data_group = parser.add_argument_group("Data and Paths")
    data_group.add_argument("--dataset_name", type=str, required=True, 
                           help="Name of the FiftyOne dataset to use.")
    data_group.add_argument("--output_dir", type=str, required=True, 
                           help="Directory to save checkpoints and logs.")
    data_group.add_argument("--train_tag", type=str, default=None, 
                           help="Dataset tag for training samples. If omitted, use full dataset.")
    data_group.add_argument("--val_tag", type=str, default="val", 
                           help="Dataset tag for validation samples. Use 'none' to disable.")
    data_group.add_argument("--num_workers", type=int, default=4, 
                           help="Number of workers for data loading.")
    data_group.add_argument("--image_size", type=int, default=224, 
                           help="Input image size. Recommended 384-448 for fine-grained.")
    data_group.add_argument("--exclude_classes", action="append", default=None,
                           help="Class labels to exclude from train/val.")
    data_group.add_argument("--cache_dir", type=str, default=None,
                           help="Directory to store dataset cache files. Defaults to ~/.cache/fish_identification")
    data_group.add_argument("--no_cache", action="store_true",
                           help="Disable dataset caching.")

    # ==========================================================================
    # Model Architecture
    # ==========================================================================
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--backbone_model_name", type=str, default="maxvit_base_tf_224",
                            help="Name of the timm backbone model.")
    model_group.add_argument("--embedding_dim", type=int, default=512,
                            help="Dimensionality of the final embedding.")
    model_group.add_argument("--pooling_type", type=str, default="attention",
                            choices=["attention", "gem", "hybrid", "avg"],
                            help="Pooling strategy: attention, gem, hybrid, or avg.")
    model_group.add_argument("--auto_backbone_defaults", type=str2bool, default=False,
                            help="Apply recommended hyperparameters based on backbone.")

    # ==========================================================================
    # ArcFace Parameters
    # ==========================================================================
    arcface_group = parser.add_argument_group("ArcFace Parameters")
    arcface_group.add_argument("--arcface_s", type=float, default=64.0,
                              help="ArcFace scale parameter 's'.")
    arcface_group.add_argument("--arcface_m", type=float, default=0.2,
                              help="ArcFace margin parameter 'm'.")

    # ==========================================================================
    # Loss Function Configuration
    # ==========================================================================
    loss_group = parser.add_argument_group("Loss Function Configuration")
    loss_group.add_argument("--loss_type", type=str, default="combined",
                           choices=["combined", "combined_v2", "focal_only"],
                           help="Loss function type. 'combined' is original, 'combined_v2' adds Focal Loss.")
    loss_group.add_argument("--metric_loss_type", type=str, default="threshold_consistent",
                           choices=["threshold_consistent", "multi_similarity", "triplet", "circle", "supcon"],
                           help="Metric learning loss type.")
    loss_group.add_argument("--miner_type", type=str, default="batch_hard",
                           choices=["batch_hard", "multi_similarity", "triplet", "none"],
                           help="Hard negative miner type.")
    loss_group.add_argument("--arcface_weight", type=float, default=0.9,
                           help="Weight for CE(ArcFace logits) in CombinedLoss.")
    loss_group.add_argument("--metric_weight", type=float, default=0.1,
                           help="Weight for metric-learning loss in CombinedLoss.")
    loss_group.add_argument("--focal_weight", type=float, default=0.0,
                           help="Weight for additional Focal Loss (for combined_v2).")
    loss_group.add_argument("--focal_gamma", type=float, default=2.0,
                           help="Gamma parameter for Focal Loss.")
    loss_group.add_argument("--focal_alpha", type=float, default=0.25,
                           help="Alpha parameter for Focal Loss.")
    loss_group.add_argument("--label_smoothing", type=float, default=0.1,
                           help="Label smoothing for CE loss.")
    loss_group.add_argument("--use_cross_batch_memory", type=str2bool, default=False,
                           help="Enable cross-batch memory for metric loss.")
    loss_group.add_argument("--memory_size", type=int, default=4096,
                           help="Size of cross-batch memory.")

    # ==========================================================================
    # Data Augmentation
    # ==========================================================================
    aug_group = parser.add_argument_group("Data Augmentation")
    aug_group.add_argument("--augmentation_preset", type=str, default="standard",
                          choices=["basic", "standard", "strong", "medium"],
                          help="Augmentation preset: basic, standard, or strong.")
    aug_group.add_argument("--use_mixup", type=str2bool, default=False,
                          help="Enable MixUp augmentation.")
    aug_group.add_argument("--mixup_alpha", type=float, default=0.4,
                          help="MixUp alpha parameter (beta distribution).")
    aug_group.add_argument("--mixup_prob", type=float, default=0.5,
                          help="Probability of applying MixUp per batch.")

    # ==========================================================================
    # Training Hyperparameters
    # ==========================================================================
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--max_epochs", type=int, default=100,
                            help="Maximum number of training epochs.")
    train_group.add_argument("--batch_size", type=int, default=32,
                            help="Batch size for validation loader.")
    train_group.add_argument("--classes_per_batch", type=int, default=24,
                            help="Number of classes per batch for MPerClassSampler.")
    train_group.add_argument("--samples_per_class", type=int, default=4,
                            help="Number of samples per class for MPerClassSampler.")
    train_group.add_argument("--accumulate_grad_batches", type=int, default=1,
                            help="Gradient accumulation steps.")
    train_group.add_argument("--learning_rate", type=float, default=1e-4,
                            help="Optimizer learning rate.")
    train_group.add_argument("--lr_eta_min", type=float, default=1e-7,
                            help="Minimum LR for cosine scheduler.")
    train_group.add_argument("--weight_decay", type=float, default=0.05,
                            help="Weight decay for AdamW optimizer.")
    train_group.add_argument("--attention_loss_lambda", type=float, default=0.15,
                            help="Weight for the attention guidance loss.")
    train_group.add_argument("--freeze_backbone_epochs", type=int, default=0,
                            help="Freeze backbone for first N epochs.")
    train_group.add_argument("--attention_warmup_epochs", type=int, default=0,
                            help="Disable attention guidance loss for first N epochs.")
    train_group.add_argument("--embedding_dropout_rate", type=float, default=None,
                            help="Optional dropout after embedding projection.")
    train_group.add_argument("--use_swa", type=str2bool, default=False,
                            help="Enable Stochastic Weight Averaging.")
    train_group.add_argument("--swa_lrs", type=float, default=1e-6,
                            help="Learning rate for SWA.")
    train_group.add_argument("--swa_epoch_start", type=float, default=0.75,
                            help="Start SWA at this fraction of training (0.75 = 75%% of epochs).")
    train_group.add_argument("--use_cyclic_lr", type=str2bool, default=False,
                            help="Use Cyclic LR or CosineAnnealingWarmRestarts instead of regular Cosine.")
    train_group.add_argument("--cyclic_mode", type=str, default="warm_restarts",
                            choices=["triangular", "triangular2", "exp_range", "warm_restarts"],
                            help="Cyclic LR mode: triangular, triangular2, exp_range, or warm_restarts (CosineAnnealingWarmRestarts).")
    train_group.add_argument("--cyclic_t0", type=int, default=10,
                            help="T_0 parameter for CosineAnnealingWarmRestarts (restart every N epochs).")

    # ==========================================================================
    # Visualization
    # ==========================================================================
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument("--visualize_attention_map", type=str2bool, default=True,
                          help="Whether to save attention map visualizations.")

    # ==========================================================================
    # Resuming and Debugging
    # ==========================================================================
    debug_group = parser.add_argument_group("Resuming and Debugging")
    debug_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                            help="Path to a checkpoint to resume training from.")
    debug_group.add_argument("--load_weights_from_checkpoint", type=str, default=None,
                            help="Load model weights from checkpoint (ignores mismatched layers).")
    debug_group.add_argument("--limit_train_batches", type=float, default=1.0,
                            help="Fraction of training data to use.")
    debug_group.add_argument("--limit_val_batches", type=float, default=1.0,
                            help="Fraction of validation data to use.")
    
    args = parser.parse_args()
    args.exclude_classes = parse_exclude_classes(args.exclude_classes)
    args.train_tag = parse_optional_tag(args.train_tag)
    args.val_tag = parse_optional_tag(args.val_tag)
    return args


def save_config(args, output_dir):
    """Saves the configuration arguments to a JSON file."""
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Configuration saved to {config_path}")


def apply_backbone_defaults(args):
    """Optionally override hyperparameters based on backbone family."""
    if not getattr(args, "auto_backbone_defaults", False):
        return

    name = str(args.backbone_model_name or "").lower()
    overrides = {}
    
    if "maxvit" in name or "maxxvit" in name:
        overrides = {
            "image_size": 384,
            "learning_rate": 5e-5,
            "freeze_backbone_epochs": 2,
            "attention_warmup_epochs": 2,
        }
    elif "convnext" in name:
        overrides = {
            "image_size": 224,
            "learning_rate": 1e-4,
            "freeze_backbone_epochs": 0,
            "attention_warmup_epochs": 0,
        }
    elif "eva" in name:
        overrides = {
            "image_size": 448 if "448" in name else 336 if "336" in name else 224,
            "learning_rate": 3e-5,
            "freeze_backbone_epochs": 3,
            "attention_warmup_epochs": 3,
        }
    elif "dino" in name:
        overrides = {
            "image_size": 518 if "518" in name else 224,
            "learning_rate": 3e-5,
            "freeze_backbone_epochs": 2,
            "attention_warmup_epochs": 2,
        }
    elif "swin" in name:
        overrides = {
            "image_size": 384 if "384" in name else 256 if "256" in name else 224,
            "learning_rate": 5e-5,
            "freeze_backbone_epochs": 1,
        }

    if not overrides:
        logger.info("Auto backbone defaults: no overrides for %s", args.backbone_model_name)
        return

    for key, new_value in overrides.items():
        old_value = getattr(args, key, None)
        if old_value != new_value:
            setattr(args, key, new_value)
            logger.info("Auto backbone defaults: %s %r -> %r", key, old_value, new_value)


def main(args):
    """Main function to set up and run the training process."""
    pl.seed_everything(42, workers=True)

    apply_backbone_defaults(args)
    
    # --- unique run identification ---
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.backbone_model_name}_{timestamp}"
    args.output_dir = os.path.join(args.output_dir, run_name)
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args, args.output_dir)
    
    # Setup cache directory (use parent of output_dir if not specified)
    if args.cache_dir is None:
        # Use parent directory of output_dir for cache
        parent_dir = os.path.dirname(args.output_dir)
        args.cache_dir = os.path.join(parent_dir, ".dataset_cache")
        logger.info(f"Cache directory set to: {args.cache_dir}")
    
    logger.info(f"Output directory set to: {args.output_dir}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Metric loss type: {args.metric_loss_type}")
    logger.info(f"Miner type: {args.miner_type}")
    logger.info(f"Pooling type: {args.pooling_type}")
    logger.info(f"Augmentation preset: {args.augmentation_preset}")
    logger.info(f"MixUp enabled: {args.use_mixup}")
    
    # 1. Setup DataModule
    datamodule = ImageEmbeddingDataModule(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
        num_workers=args.num_workers,
        exclude_classes=args.exclude_classes,
        train_tag=args.train_tag,
        val_tag=args.val_tag,
        augmentation_preset=args.augmentation_preset,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    datamodule.setup()

    # Persist label mapping
    id_to_label = {idx: label for label, idx in datamodule.label_to_id.items()}
    labels_path = os.path.join(args.output_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(id_to_label, f, indent=2, ensure_ascii=False)
    logger.info("Saved id_to_label mapping to %s", labels_path)
    
    # 2. Setup Model
    trainer_cls = ImageEmbeddingTrainerConvnext if 'convnext' in args.backbone_model_name.lower() else ImageEmbeddingTrainerViT
    
    model = trainer_cls(
        num_classes=datamodule.num_classes,
        embedding_dim=args.embedding_dim,
        backbone_model_name=args.backbone_model_name,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        arcface_weight=args.arcface_weight,
        metric_weight=args.metric_weight,
        label_smoothing=args.label_smoothing,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_eta_min=args.lr_eta_min,
        attention_loss_lambda=args.attention_loss_lambda,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        attention_warmup_epochs=args.attention_warmup_epochs,
        embedding_dropout_rate=args.embedding_dropout_rate,
        load_checkpoint=args.load_weights_from_checkpoint,
        output_dir=args.output_dir,
        visualize_attention_map=args.visualize_attention_map,
        # New parameters
        loss_type=args.loss_type,
        metric_loss_type=args.metric_loss_type,
        miner_type=args.miner_type,
        use_cross_batch_memory=args.use_cross_batch_memory,
        memory_size=args.memory_size,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        focal_weight=args.focal_weight,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        pooling_type=args.pooling_type,
        # Cyclic LR parameters
        use_cyclic_lr=args.use_cyclic_lr,
        cyclic_mode=args.cyclic_mode,
        cyclic_t0=args.cyclic_t0,
        max_epochs=args.max_epochs,
    )
    
    if args.load_weights_from_checkpoint:
        stats = getattr(model, "_loaded_weight_stats", None)
        if stats:
            logger.info(
                "Loaded weights from checkpoint: %d/%d tensors matched "
                "(skipped mismatched: %d, missing: %d, unexpected: %d).",
                stats["loaded"],
                stats["total_in_checkpoint"],
                stats["skipped_mismatch"],
                stats["missing_in_model"],
                stats["unexpected_in_checkpoint"],
            )
        else:
            logger.info("Loaded weights from checkpoint: stats not available.")
    
    # 3. Setup Callbacks and Logger
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tensorboard_logger = TensorBoardLogger(save_dir=args.output_dir, name="logs")
    metrics_file_logger = MetricsToFileCallback(output_dir=args.output_dir)

    has_val = args.val_tag is not None and datamodule.val_dataset is not None
    if has_val:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/accuracy_epoch",
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="model-{epoch:02d}-{val/accuracy_epoch:.4f}",
            save_top_k=3,
            mode="max",
            save_last=True,
            save_weights_only=True  # Save only model weights, not optimizer states
        )
        # Disable early stopping when using SWA (needs full training)
        if not args.use_swa:
            early_stopping = EarlyStopping(
                monitor="val/accuracy_epoch",
                patience=15,
                mode="max",
                verbose=True
            )
            callbacks = [checkpoint_callback, lr_monitor, early_stopping, metrics_file_logger]
        else:
            logger.info("SWA enabled: disabling early stopping to allow full training")
            callbacks = [checkpoint_callback, lr_monitor, metrics_file_logger]
        limit_val_batches = args.limit_val_batches
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            save_top_k=1,
            save_last=True,
            save_weights_only=True  # Save only model weights, not optimizer states
        )
        callbacks = [checkpoint_callback, lr_monitor, metrics_file_logger]
        limit_val_batches = 0.0
    
    # Add SWA callback if enabled
    if args.use_swa:
        logger.info(f"Enabling Stochastic Weight Averaging (SWA) with lr={args.swa_lrs}, starting at {args.swa_epoch_start*100:.0f}% of training")
        swa_callback = StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=10,
            annealing_strategy="cos",
            device=None  # Will use model's device
        )
        callbacks.append(swa_callback)

    # 4. Setup Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        max_epochs=args.max_epochs,
        logger=tensorboard_logger,
        callbacks=callbacks,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=False
    )

    # 5. Start Training
    logger.info("Starting model training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )

    # 6. Post-training evaluation
    def _resolve_checkpoint_for_eval() -> str:
        best = getattr(checkpoint_callback, "best_model_path", None)
        if best and os.path.exists(best):
            return best
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt
        if os.path.isdir(ckpt_dir):
            ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
            if ckpts:
                ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return ckpts[0]
        raise FileNotFoundError(f"Could not find a checkpoint to evaluate in: {ckpt_dir}")

    try:
        ckpt_path = _resolve_checkpoint_for_eval()
        post_eval_root = os.path.join(args.output_dir, "posttrain_eval")
        os.makedirs(post_eval_root, exist_ok=True)

        eval_script = os.path.join(
            CURRENT_FOLDER_PATH[:pos + len(DELIMITER)],
            "train_scripts",
            "classification",
            "posttrain_eval.py",
        )
        cmd = [
            sys.executable,
            eval_script,
            "--dataset_name",
            args.dataset_name,
            "--checkpoint",
            ckpt_path,
            "--output_dir",
            post_eval_root,
            "--backbone_model_name",
            args.backbone_model_name,
            "--embedding_dim",
            str(args.embedding_dim),
            "--arcface_s",
            str(args.arcface_s),
            "--arcface_m",
            str(args.arcface_m),
            "--batch_size",
            "64",
        ]
        if getattr(args, "image_size", None) is not None:
            cmd += ["--image_size", str(args.image_size)]

        logger.info("Running posttrain eval: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Posttrain eval completed. Results are in: %s", post_eval_root)
    except Exception as e:
        logger.warning("Posttrain eval failed: %s", e)

    logger.info("Training finished successfully.")


if __name__ == "__main__":
    cli_args = get_args()
    main(cli_args)
