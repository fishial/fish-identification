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

# Training with GeM pooling
python lightning_train.py \\
    --dataset_name "my_fiftyone_dataset" \\
    --output_dir "/path/to/experiments" \\
    --pooling_type gem \\
    --backbone_model_name convnext_base

# Using EVA-02 backbone with strong augmentations
python lightning_train.py \\
    --dataset_name "my_fiftyone_dataset" \\
    --output_dir "/path/to/experiments" \\
    --backbone_model_name eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \\
    --image_width 448 --image_height 448 \\
    --augmentation_preset strong \\
    --loss_type combined_v2
"""

# --- Standard Library Imports ---

# --- Third-Party Imports ---
import argparse
import logging
import math
import os
import random
import sys
import subprocess
import json
from typing import Optional, Dict, List

# --- project root helper ----------------------------------------------------
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = "fish-identification"
POS = CURRENT_FOLDER_PATH.find(DELIMITER)
if POS != -1:
    BASE_PATH = CURRENT_FOLDER_PATH[:POS + len(DELIMITER)]
    if BASE_PATH not in sys.path:
        sys.path.insert(1, BASE_PATH)
        print("SETUP: sys.path updated")

# --- Third-Party Imports ---
import numpy as np

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
from module.classification_package.src.lightning_trainer import (
    ImageEmbeddingTrainerViT,
)
from module.classification_package.src.loss_functions import compute_class_weights
from module.classification_package.src.data.module.v2 import ImageEmbeddingDataModule
from module.classification_package.src.host_memory_monitor import HostMemoryMonitorCallback
from module.classification_package.src.auto_tuner import (
    AutoTuner,
    AdaptiveRegularizationCallback,
    generate_next_run_suggestion,
    resolve_run_dir,
    find_best_checkpoint,
)

import csv

# Setup professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetricsToFileCallback(pl.Callback):
    """
    Writes callback_metrics at the end of each validation epoch to:
      - metrics_history.jsonl (raw row-by-row data)
      - metrics_history.csv (raw row-by-row data)
      - metrics_summary.json (consolidated arrays rounded to 3 decimals)
    inside the current run directory.
    """
    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "metrics_history.jsonl")
        self.csv_path = os.path.join(output_dir, "metrics_history.csv")
        self.json_summary_path = os.path.join(output_dir, "metrics_summary.json")
        
        self._csv_header_written = False
        self._csv_keys: List[str] = []

        # Data structure for the final JSON summary
        self.summary_data = {
            'epoch': [],
            'lr': [],                   # <-- NEW
            'train_acc': [],
            'train_loss': [],
            'train_loss_main': [],      # <-- NEW
            'train_loss_div': [],       # <-- NEW
            'train_loss_cov': [],       # <-- NEW
            'train_loss_guide': [],     # <-- NEW
            'train_loss_seg': [],
            'val_acc': [],
            'val_acc_top5': [],
            'val_macro_acc': [],
            'val_loss': [],
            'val_fisher_ratio': [],
            'val_intra_var': [],
            'val_inter_dist': []
        }

        # Mapping: "Key in our JSON" -> "Key in Lightning logs"
        self.metric_mapping = {
            'epoch': 'epoch',
            'lr': 'train/lr',                             # <-- NEW (make sure you log it exactly like this)
            'train_acc': 'train/accuracy_epoch', 
            'train_loss': 'train/loss_epoch',
            'train_loss_main': 'train/loss_main',         # <-- NEW
            'train_loss_div': 'train/loss_diversity',     # <-- NEW
            'train_loss_cov': 'train/loss_coverage',      # <-- NEW
            'train_loss_guide': 'train/loss_attn_guidance', # <-- NEW
            'train_loss_seg': 'train/loss_seg',   # <-- ADD THIS
            'val_acc': 'val/accuracy_epoch',
            'val_acc_top5': 'val/accuracy_top5_epoch',
            'val_macro_acc': 'val/accuracy_macro_epoch',
            'val_loss': 'val/loss',
            'val_fisher_ratio': 'val/fisher_ratio',
            'val_intra_var': 'val/intra_var',
            'val_inter_dist': 'val/inter_dist'
        }

        # If the summary file already exists (e.g., resuming training), load it
        if os.path.exists(self.json_summary_path):
            try:
                with open(self.json_summary_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    # Ensure the structure matches
                    if all(k in loaded_data for k in self.summary_data.keys()):
                        self.summary_data = loaded_data
            except Exception as e:
                print(f"Warning: Could not load existing summary JSON. Starting fresh. Error: {e}")

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

    @staticmethod
    def _round_val(v):
        """Round floating-point numbers to three decimal places"""
        if isinstance(v, float):
            return round(v, 3)
        return v

    def _write_csv_row(self, row: dict):
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
        
        # 1. Assemble the raw row for CSV and JSONL
        row = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
        }
        for k, v in metrics.items():
            if k.startswith("hp_metric"):
                continue
            row[k] = self._to_py(v)

        os.makedirs(self.output_dir, exist_ok=True)
        
        # Write the raw data (JSONL and CSV)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._write_csv_row(row)

        # 2. Update the consolidated summary JSON
        for summary_key, pl_key in self.metric_mapping.items():
            # Fetch the value; use None if missing (keeps all lists equal length)
            val = row.get(pl_key, None)
            rounded_val = self._round_val(val)
            self.summary_data[summary_key].append(rounded_val)

        # Overwrite the summary file completely (it always stores full arrays)
        with open(self.json_summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_data, f, indent=4, ensure_ascii=False)


# =================================================================================
# ARGUMENT PARSING
# =================================================================================

def get_args():
    """Parses command-line arguments."""
    # ------------------------------------------------------------------
    # Pre-pass: extract --config before building the full parser so that
    # values from the JSON file can be used as defaults and any explicit
    # CLI flags still take precedence.
    # ------------------------------------------------------------------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="Path to a config.json file. Values in the file "
                                 "become defaults; explicit CLI flags override them.")
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults: dict = {}
    if pre_args.config:
        config_path = os.path.abspath(pre_args.config)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as _f:
            config_defaults = json.load(_f)
        logger.info("Loaded config from: %s", config_path)

    parser = argparse.ArgumentParser(
        description="Train an Image Embedding Model with configurable loss and augmentations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load all settings from a saved config.json
  python lightning_train.py --config /path/to/config.json

  # Load from config, but override a specific parameter
  python lightning_train.py --config /path/to/config.json --max_epochs 50

  # Basic training (backward compatible)
  python lightning_train.py --dataset_name my_dataset --output_dir ./experiments

  # With improved loss function
  python lightning_train.py --dataset_name my_dataset --output_dir ./experiments \\
      --loss_type combined_v2 --metric_loss_type multi_similarity --focal_weight 0.1

  # With strong augmentations
  python lightning_train.py --dataset_name my_dataset --output_dir ./experiments \\
      --augmentation_preset strong
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

    def parse_optional_tag(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if value.lower() in {"none", "null", ""}:
            return None
        return value

    # ==========================================================================
    # Config file (already handled in pre-pass above; kept here for --help)
    # ==========================================================================
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="Path to a config.json file. All parameters in the file "
                             "become defaults; explicit CLI flags still override them.")

    # ==========================================================================
    # Data and Paths
    # ==========================================================================
    data_group = parser.add_argument_group("Data and Paths")
    data_group.add_argument("--dataset_name", type=str, default=None,
                           help="Name of the FiftyOne dataset to use.")
    data_group.add_argument("--output_dir", type=str, default=None,
                           help="Directory to save checkpoints and logs.")
    data_group.add_argument("--train_tag", type=str, default=None, 
                           help="Dataset tag for training samples. If omitted, use full dataset.")
    data_group.add_argument("--val_tag", type=str, default="val", 
                           help="Dataset tag for validation samples. Use 'none' to disable.")
    data_group.add_argument("--num_workers", type=int, default=8,
                           help="Training DataLoader workers (match ImageEmbeddingDataModule default before wiring; lower if CPU/RAM bound).")
    data_group.add_argument("--val_num_workers", type=int, default=2,
                           help="Workers for validation only (lower = less RAM; full-res cv2 reads).")
    data_group.add_argument("--val_batch_size", type=int, default=32,
                           help="Validation batch size (lower if validation causes OOM).")
    data_group.add_argument("--log_host_memory", type=str2bool, default=True,
                           help="Log host RSS (trainer + DataLoader workers) each epoch; writes host_memory_log.jsonl.")
    data_group.add_argument("--image_width", type=int, default=518,
                           help="Input image width in pixels (default: 518).")
    data_group.add_argument("--image_height", type=int, default=154,
                           help="Input image height in pixels (default: 154).")
    data_group.add_argument("--cache_dir", type=str, default="/home/andrew/Andrew/Fishial2402/Experiments/v11/grid_search_20260228_081752",
                           help="Directory to store dataset cache files. Defaults to ~/.cache/fish_identification")
    data_group.add_argument("--no_cache", action="store_true",
                           help="Disable dataset caching.")
    data_group.add_argument("--class_mapping_path", type=str, default=None,
                           help="Path to class mapping file.")
    data_group.add_argument("--bg_removal_prob", type=float, default=0.0,
                           help="Probability of removing background.")
    data_group.add_argument("--bbox_padding_limit", type=float, default=0.15,
                           help="Limit for bbox padding.")
    data_group.add_argument("--max_samples_per_class", type=int, default=100,
                           help="Maximum number of samples per class.")
    data_group.add_argument("--instance_data", type=str2bool, default=False,
                           help="Use instance data.")
    data_group.add_argument("--labels_path", type=str, default=None,
                           help="Path to selected labels file (e.g. labels_150.txt).")
    # ==========================================================================
    # Model Architecture
    # ==========================================================================
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--backbone_model_name", type=str, default="maxvit_base_tf_224",
                            help="Name of the timm backbone model.")
    model_group.add_argument("--embedding_dim", type=int, default=512,
                            help="Dimensionality of the final embedding.")
    model_group.add_argument("--pooling_type", type=str, default="attention",
                            choices=["attention", "attention_single_head", "gem", "hybrid", "avg", "mean"],
                            help="Pooling strategy: attention, gem, hybrid, or avg.")
    model_group.add_argument("--num_attention_heads", type=int, default=4,
                            help="Number of attention heads for multi-head pooling (ViT only).")
    model_group.add_argument("--use_cls_token", type=str2bool, default=True,
                            help="Fuse CLS token with attention-pooled patches (ViT only).")
    model_group.add_argument("--auto_backbone_defaults", type=str2bool, default=False,
                            help="Apply recommended hyperparameters based on backbone.")

    model_group.add_argument("--neck_type", type=str, default="simple",
                            choices=["simple", "mlp", "advanced", "bnneck", "lnneck", "resneck", "gated"],
                            help="Neck type: simple or mlp.")
    model_group.add_argument("--head_type", type=str, default="arcface",
                            choices=["arcface", "subcenter"],
                            help="Head type: arcface or subcenter.")    
    model_group.add_argument("--train_segmentation_head", type=str2bool, default=False,
                            help="Freeze everything except the segmentation head.")
    # ==========================================================================
    # ArcFace Parameters
    # ==========================================================================
    arcface_group = parser.add_argument_group("ArcFace Parameters")
    arcface_group.add_argument("--arcface_s", type=float, default=64.0,
                              help="ArcFace scale parameter 's'.")
    arcface_group.add_argument("--arcface_m", type=float, default=0.2,
                              help="ArcFace margin parameter 'm'.")
    arcface_group.add_argument("--arcface_K", type=int, default=3,
                              help="Number of sub-centers per class (SubCenter ArcFace). "
                                   "K>1 handles intra-class variation (pose, dimorphism). Default: 3.")
    arcface_group.add_argument("--use_adacos", type=str2bool, default=False,
                              help="Enable AdaCos adaptive scale (auto-tunes s during training). "
                                   "Recommended for medium-sized class sets (100-2000 classes).")
    arcface_group.add_argument("--use_dynamic_margin", type=str2bool, default=False,
                              help="Ramp ArcFace margin from arcface_m_start to arcface_m over 70%% of training.")
    arcface_group.add_argument("--arcface_m_start", type=float, default=0.1,
                              help="Starting margin when use_dynamic_margin is enabled.")
    arcface_group.add_argument("--use_class_weights", type=str2bool, default=False,
                              help="Use class weights for the loss function.")
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
    loss_group.add_argument("--max_fisher_val_samples", type=int, default=8192,
                           help="Cap CPU embeddings for Fisher metric during val (-1 = no cap).")

    # ==========================================================================
    # Data Augmentation
    # ==========================================================================
    aug_group = parser.add_argument_group("Data Augmentation")
    aug_group.add_argument("--augmentation_preset", type=str, default="standard",
                          choices=["basic", "standard", "strong", "medium"],
                          help="Augmentation preset: basic, standard, or strong.")
    aug_group.add_argument("--resize_strategy", type=str, default="pad",
                          choices=["pad", "squish"],
                          help="Resize strategy: pad or squish.")
    aug_group.add_argument("--alignment_method", type=str, default="diagonal",
                          choices=["diagonal", "horizontal"],
                          help="Alignment method: diagonal or horizontal.")
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
    train_group.add_argument("--coverage_loss_lambda", type=float, default=0.1,
                            help="Weight for the coverage loss.")
    train_group.add_argument("--diversity_loss_lambda", type=float, default=0.2,
                            help="Weight for the diversity loss.")
    train_group.add_argument("--segmentation_loss_lambda", type=float, default=1.0,
                            help="Weight for the segmentation loss.")

    train_group.add_argument("--freeze_backbone_epochs", type=int, default=0,
                            help="Freeze backbone for first N epochs.")
    train_group.add_argument("--attention_warmup_epochs", type=int, default=0,
                            help="Disable attention guidance loss for first N epochs.")
    train_group.add_argument("--embedding_dropout_rate", type=float, default=None,
                            help="Optional dropout after embedding projection.")
    train_group.add_argument("--drop_path_rate", type=float, default=0.0,
                            help="Stochastic depth rate for backbone ViT blocks (0.1 recommended for fine-tuning).")
    train_group.add_argument("--gradient_clip_val", type=float, default=1.0,
                            help="Max norm for gradient clipping (0 = disabled).")
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
    viz_group.add_argument("--visualize_attention_map", type=str2bool, default=False,
                          help="Whether to save attention map visualizations.")

    # ==========================================================================
    # Resuming and Debugging
    # ==========================================================================
    debug_group = parser.add_argument_group("Resuming and Debugging")
    debug_group.add_argument("--validate_only", action="store_true",
                            help="Run only the validation loop (requires --load_weights_from_checkpoint).")
    debug_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                            help="Path to a checkpoint to resume training from.")
    debug_group.add_argument("--load_weights_from_checkpoint", type=str, default=None,
                            help="Load model weights from checkpoint (ignores mismatched layers).")
    debug_group.add_argument("--limit_train_batches", type=float, default=1.0,
                            help="Fraction of training data to use.")
    debug_group.add_argument("--limit_val_batches", type=float, default=1.0,
                            help="Fraction of validation data to use.")

    # ==========================================================================
    # Auto-Tuning
    # ==========================================================================
    tune_group = parser.add_argument_group("Auto-Tuning")
    tune_group.add_argument("--auto_tune_from", type=str, default=None,
                           help="Path to a previous run directory (or its parent to "
                                "auto-pick latest). Analyses metrics_history.jsonl and "
                                "adjusts hyperparameters to counter overfitting / "
                                "underfitting / stagnation.")
    tune_group.add_argument("--use_adaptive_regularization", type=str2bool, default=False,
                           help="Enable adaptive regularization callback that dynamically "
                                "increases dropout and weight decay when overfitting is "
                                "detected during training.")
    tune_group.add_argument("--adaptive_gap_threshold", type=float, default=0.025,
                           help="Train-val accuracy gap threshold for adaptive reg.")
    tune_group.add_argument("--adaptive_patience", type=int, default=3,
                           help="Consecutive checks above threshold before adjusting.")

    # Apply JSON config values as defaults (CLI flags override them).
    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    args.train_tag = parse_optional_tag(args.train_tag)
    args.val_tag = parse_optional_tag(args.val_tag)

    # Validate required fields that may have come from the config file.
    missing = [f for f in ("dataset_name", "output_dir") if not getattr(args, f, None)]
    if missing:
        parser.error(
            f"The following arguments are required (either via CLI or --config): "
            + ", ".join(f"--{m}" for m in missing)
        )

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
            "image_width": 384,
            "image_height": 384,
            "learning_rate": 5e-5,
            "freeze_backbone_epochs": 2,
            "attention_warmup_epochs": 2,
        }
    elif "convnext" in name:
        overrides = {
            "image_width": 224,
            "image_height": 224,
            "learning_rate": 1e-4,
            "freeze_backbone_epochs": 0,
            "attention_warmup_epochs": 0,
        }
    elif "eva" in name:
        _s = 448 if "448" in name else 336 if "336" in name else 224
        overrides = {
            "image_width": _s,
            "image_height": _s,
            "learning_rate": 3e-5,
            "freeze_backbone_epochs": 3,
            "attention_warmup_epochs": 3,
        }
    elif "dino" in name:
        overrides = {
            # "image_size": 392,
            # "learning_rate": 3e-5,
            # "freeze_backbone_epochs": 0,
            # "attention_warmup_epochs": 0,
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

    # --- Auto-tune from previous run ---
    if args.auto_tune_from:
        try:
            run_dir = resolve_run_dir(args.auto_tune_from)
            tuner = AutoTuner(run_dir)
            tuner.apply_to_args(args)
        except Exception as e:
            logger.warning("[AutoTuner] Failed to apply auto-tuning: %s", e)

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
        parent_dir = "/home/andrew/Andrew/Fishial2402/Experiments/v11"
        args.cache_dir = os.path.join(parent_dir, ".dataset_cache")
        logger.info(f"Cache directory set to: {args.cache_dir}")
    
    logger.info(f"Output directory set to: {args.output_dir}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Metric loss type: {args.metric_loss_type}")
    logger.info(f"Miner type: {args.miner_type}")
    logger.info(f"Pooling type: {args.pooling_type}")
    logger.info(f"Augmentation preset: {args.augmentation_preset}")
    logger.info(f"Resize strategy: {args.resize_strategy}")

    # 1. Setup DataModule
    train_tags = [f"train_{i}" for i in range(0, args.max_samples_per_class or 1e5+1)]
    datamodule = ImageEmbeddingDataModule(
        dataset_name = args.dataset_name,
        labels_path = args.labels_path,
        class_mapping_path= args.class_mapping_path,
        classes_per_batch = args.classes_per_batch,
        samples_per_class = args.samples_per_class,
        image_size = (args.image_height, args.image_width),
        train_tags =train_tags,
        val_tags = [args.val_tag],
        augmentation_preset = args.augmentation_preset,
        instance_data = args.instance_data  ,
        cache_dir = args.cache_dir,
        use_cache = not args.no_cache,
        bg_removal_prob = args.bg_removal_prob,
        bbox_padding_limit = args.bbox_padding_limit,
        alignment_method = args.alignment_method,
        resize_strategy = args.resize_strategy,
        num_workers=args.num_workers,
        val_batch_size=args.val_batch_size,
        val_num_workers=args.val_num_workers,
    )
    datamodule.setup()

    # Compute per-class weights for focal loss (handles class imbalance)
    class_weights = None
    if hasattr(datamodule, 'train_dataset') and datamodule.train_dataset is not None and args.use_class_weights:
        try:
            class_weights = compute_class_weights(
                datamodule.train_dataset.targets, datamodule.num_classes
            )
            logger.info(
                "Computed class weights: min=%.4f, max=%.4f, mean=%.4f",
                class_weights.min().item(), class_weights.max().item(), class_weights.mean().item(),
            )
        except Exception as e:
            logger.warning("Failed to compute class weights: %s", e)

    # 2. Setup Model
    trainer_cls = ImageEmbeddingTrainerViT
    
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
        coverage_loss_lambda=args.coverage_loss_lambda,
        diversity_loss_lambda=args.diversity_loss_lambda,
        segmentation_loss_lambda=args.segmentation_loss_lambda,

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
        max_fisher_val_samples=args.max_fisher_val_samples,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        focal_weight=args.focal_weight,
        pooling_type=args.pooling_type,
        num_attention_heads=args.num_attention_heads,
        use_cls_token=args.use_cls_token,
        # SubCenter ArcFace / AdaCos
        arcface_K=args.arcface_K,
        use_adacos=args.use_adacos,
        # Dynamic ArcFace margin
        use_dynamic_margin=args.use_dynamic_margin,
        arcface_m_start=args.arcface_m_start,
        # Cyclic LR parameters
        use_cyclic_lr=args.use_cyclic_lr,
        cyclic_mode=args.cyclic_mode,
        cyclic_t0=args.cyclic_t0,
        max_epochs=args.max_epochs,
        neck_type=args.neck_type,
        head_type=args.head_type,
        input_img_size=[args.image_height, args.image_width],

        # Parameters for segmentation
        train_segmentation_head=args.train_segmentation_head,

        # Backbone regularization
        drop_path_rate=args.drop_path_rate,

        # Per-class weights for focal loss
        class_weights=class_weights,
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
    memory_cb = HostMemoryMonitorCallback(
        output_dir=args.output_dir,
        enabled=args.log_host_memory,
    )

    has_val = args.val_tag is not None and datamodule.val_dataset is not None
    if has_val:
        checkpoint_callback = ModelCheckpoint(
            monitor="val/accuracy_epoch",
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="model-epoch{epoch:02d}-acc{val_acc_int:.0f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode="max",
            save_last=True,
            save_weights_only=True,
        )
        if not args.use_swa:
            early_stopping = EarlyStopping(
                monitor="val/accuracy_epoch",
                patience=15,
                mode="max",
                verbose=True
            )
            callbacks = [checkpoint_callback, lr_monitor, metrics_file_logger, memory_cb]
        else:
            logger.info("SWA enabled: disabling early stopping to allow full training")
            callbacks = [checkpoint_callback, lr_monitor, metrics_file_logger, memory_cb]
        limit_val_batches = args.limit_val_batches
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            save_top_k=1,
            save_last=True,
            save_weights_only=True  # Save only model weights, not optimizer states
        )
        callbacks = [checkpoint_callback, lr_monitor, metrics_file_logger, memory_cb]
        limit_val_batches = 0.0
    
    # Add adaptive regularization callback if enabled
    if args.use_adaptive_regularization:
        adaptive_cb = AdaptiveRegularizationCallback(
            gap_threshold=args.adaptive_gap_threshold,
            patience=args.adaptive_patience,
        )
        callbacks.append(adaptive_cb)
        logger.info(
            "Adaptive regularization enabled (gap_threshold=%.3f, patience=%d)",
            args.adaptive_gap_threshold, args.adaptive_patience,
        )

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
    grad_clip = args.gradient_clip_val if args.gradient_clip_val > 0 else None
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
        gradient_clip_val=grad_clip,
        gradient_clip_algorithm="norm",
        deterministic=False,
    )

    # 5. Start Training or Validation
    if getattr(args, "validate_only", False):
        logger.info("🚀 Starting VALIDATION ONLY mode...")
        if not args.load_weights_from_checkpoint and not args.resume_from_checkpoint:
            logger.warning("⚠️ Running validation without providing a checkpoint! The model will use random initialized weights.")
        
        # Run validation only
        trainer.validate(model, datamodule=datamodule)
        
        logger.info("✅ Validation finished successfully.")
        return  # Exit main since training and post-eval are unnecessary
        
    else:
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
            CURRENT_FOLDER_PATH[:POS + len(DELIMITER)],
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
        if getattr(args, "image_width", None) is not None:
            cmd += ["--image_width", str(args.image_width)]
        if getattr(args, "image_height", None) is not None:
            cmd += ["--image_height", str(args.image_height)]

        logger.info("Running posttrain eval: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Posttrain eval completed. Results are in: %s", post_eval_root)
    except Exception as e:
        logger.warning("Posttrain eval failed: %s", e)

    # 7. Generate next-run suggestion with auto-tuned parameters
    try:
        generate_next_run_suggestion(args, args.output_dir)
    except Exception as e:
        logger.warning("Failed to generate next-run suggestion: %s", e)

    logger.info("Training finished successfully.")


if __name__ == "__main__":
    cli_args = get_args()
    main(cli_args)
