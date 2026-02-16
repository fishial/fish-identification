# -*- coding: utf-8 -*-
"""
PyTorch Lightning Trainers for Fish Classification.

This module provides Lightning modules for training image embedding models
with support for:
- Multiple backbone architectures (ViT, ConvNeXt, etc.)
- Multiple loss functions (Combined, CombinedV2 with Focal, etc.)
- Attention guidance for improved localization
- MixUp data augmentation
- Configurable pooling strategies

Key Classes:
- BaseImageEmbeddingTrainer: Base class with shared functionality
- ImageEmbeddingTrainerViT: For Vision Transformer backbones
- ImageEmbeddingTrainerConvnext: For CNN backbones
"""

import os
import random
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image

import lightning.pytorch as pl
from torchmetrics.classification import Accuracy
import torchvision.utils as vutils

from module.classification_package.src.model_v2 import StableEmbeddingModelViT, StableEmbeddingModel
from module.classification_package.src.loss_functions import (
    CombinedLoss, 
    CombinedLossV2, 
    create_loss_function,
    mixup_data,
    mixup_criterion,
)
from module.classification_package.src.visualize_utils import save_attention_overlay


class BaseImageEmbeddingTrainer(pl.LightningModule):
    """
    Base PyTorch Lightning module for training image embedding models.
    
    This base class provides shared functionality:
    - Loss function setup
    - Metrics tracking
    - Optimizer and scheduler configuration
    - Validation and visualization logic
    - Backbone freezing/unfreezing
    """
    
    def __init__(self):
        super().__init__()
        self.main_loss_fn = None
        self.attention_guidance_loss_fn = None
        self.train_accuracy = None
        self.train_accuracy_top5 = None
        self.train_accuracy_top10 = None
        self.val_accuracy = None
        self.val_accuracy_top5 = None
        self.val_accuracy_top10 = None
        self.val_accuracy_macro = None
        self.visualization_indices = None
        self._backbone_frozen = False

    def _get_attention_lambda(self) -> float:
        """Get attention loss weight, considering warmup."""
        warmup = int(getattr(self.hparams, "attention_warmup_epochs", 0) or 0)
        if self.current_epoch < warmup:
            return 0.0
        return float(getattr(self.hparams, "attention_loss_lambda", 0.0) or 0.0)

    def _setup_common_components(self, num_classes: int, embedding_dim: int = 512):
        """
        Initialize losses and metrics common to all trainers.
        
        Args:
            num_classes: Number of classes for metrics
            embedding_dim: Embedding dimension (for some loss types)
        """
        # Get loss configuration from hparams
        loss_type = getattr(self.hparams, "loss_type", "combined")
        metric_loss_type = getattr(self.hparams, "metric_loss_type", "threshold_consistent")
        miner_type = getattr(self.hparams, "miner_type", "batch_hard")
        use_cross_batch_memory = getattr(self.hparams, "use_cross_batch_memory", False)
        memory_size = getattr(self.hparams, "memory_size", 4096)
        focal_gamma = getattr(self.hparams, "focal_gamma", 2.0)
        focal_alpha = getattr(self.hparams, "focal_alpha", 0.25)
        focal_weight = getattr(self.hparams, "focal_weight", 0.0)
        
        # Create loss function using factory
        self.main_loss_fn = create_loss_function(
            loss_type=loss_type,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            arcface_weight=float(self.hparams.arcface_weight),
            metric_weight=float(self.hparams.metric_weight),
            focal_weight=focal_weight,
            label_smoothing=float(self.hparams.label_smoothing),
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            metric_loss_type=metric_loss_type,
            miner_type=miner_type,
            use_cross_batch_memory=use_cross_batch_memory,
            memory_size=memory_size,
        )
        
        # Attention guidance loss
        self.attention_guidance_loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy_macro = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        
        top5 = min(5, num_classes)
        top10 = min(10, num_classes)
        self.train_accuracy_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=top5)
        self.train_accuracy_top10 = Accuracy(task="multiclass", num_classes=num_classes, top_k=top10)
        self.val_accuracy_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=top5)
        self.val_accuracy_top10 = Accuracy(task="multiclass", num_classes=num_classes, top_k=top10)

    def _load_weights(self, path: str):
        """Load weights from a checkpoint with shape mismatch handling."""
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model_state = self.state_dict()
        
        filtered = {}
        mismatched = []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                mismatched.append(k)

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        total_in_ckpt = len(state_dict)
        loaded_count = len(filtered)
        
        self._loaded_weight_stats = {
            "total_in_checkpoint": total_in_ckpt,
            "loaded": loaded_count,
            "skipped_mismatch": len(mismatched),
            "missing_in_model": len(missing),
            "unexpected_in_checkpoint": len(unexpected),
        }
        
        if missing:
            print(f"[checkpoint] Missing keys (up to 10): {missing[:10]}")
        if unexpected:
            print(f"[checkpoint] Unexpected keys (up to 10): {unexpected[:10]}")
        if mismatched:
            print(f"[checkpoint] Mismatched keys skipped (up to 10): {mismatched[:10]}")

    def forward(self, x, labels=None, object_mask=None):
        """Forward pass through the model."""
        return self.model(
            x, 
            labels=labels, 
            object_mask=object_mask, 
            return_softmax=(not self.training),
            return_attention_map=True
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        from timm.scheduler import CosineLRScheduler
        from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Check if cyclic LR is enabled
        use_cyclic_lr = getattr(self.hparams, "use_cyclic_lr", False)
        
        if use_cyclic_lr:
            cyclic_mode = getattr(self.hparams, "cyclic_mode", "warm_restarts")
            
            if cyclic_mode == "warm_restarts":
                # CosineAnnealingWarmRestarts
                cyclic_t0 = getattr(self.hparams, "cyclic_t0", 10)
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=cyclic_t0,
                    T_mult=2,
                    eta_min=self.hparams.lr_eta_min
                )
                interval = "epoch"
                print(f"[LR Scheduler] Using CosineAnnealingWarmRestarts with T_0={cyclic_t0}")
            else:
                # CyclicLR
                max_epochs = getattr(self.hparams, "max_epochs", self.trainer.max_epochs)
                # Estimate steps per epoch (approximate)
                steps_per_epoch = 100  # Will be adjusted automatically
                scheduler = CyclicLR(
                    optimizer,
                    base_lr=self.hparams.lr_eta_min,
                    max_lr=self.hparams.lr,
                    step_size_up=steps_per_epoch * 5,
                    mode=cyclic_mode,
                    cycle_momentum=False
                )
                interval = "step"
                print(f"[LR Scheduler] Using CyclicLR with mode={cyclic_mode}")
                
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": interval,
                    "frequency": 1
                }
            }
        else:
            # Default: Cosine scheduler from timm
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.trainer.max_epochs,
                lr_min=self.hparams.lr_eta_min,
                warmup_t=5,
                warmup_lr_init=1e-6,
                cycle_limit=1
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        """Custom hook for timm and PyTorch schedulers."""
        from timm.scheduler import CosineLRScheduler
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
        
        # For timm CosineLRScheduler - needs epoch argument
        if isinstance(scheduler, CosineLRScheduler):
            scheduler.step(epoch=self.current_epoch)
        # For CosineAnnealingWarmRestarts - step per epoch
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        # For CyclicLR - do nothing, it steps per batch automatically
        elif isinstance(scheduler, CyclicLR):
            pass  # CyclicLR steps automatically in training_step
        else:
            # Fallback for other schedulers
            if hasattr(scheduler, 'step'):
                scheduler.step()

    def on_fit_start(self):
        """Called at the beginning of fit."""
        seed = 42
        pl.seed_everything(seed, workers=True)
        
        count_random_img = 63
        self.visualization_indices = set()
        if hasattr(self.trainer.datamodule, "val_dataloader"):
            val_loader = self.trainer.datamodule.val_dataloader()
            if val_loader is not None and getattr(val_loader, "dataset", None) is not None:
                total_val_size = len(val_loader.dataset)
                count_to_sample = min(count_random_img, total_val_size)
                if count_to_sample > 0:
                    self.visualization_indices = set(random.sample(range(total_val_size), count_to_sample))

        # Backbone freeze
        freeze_epochs = int(getattr(self.hparams, "freeze_backbone_epochs", 0) or 0)
        if freeze_epochs > 0:
            try:
                self.model.freeze_backbone()
                self._backbone_frozen = True
            except Exception as e:
                print(f"[freeze] Failed to freeze backbone: {e}")

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        freeze_epochs = int(getattr(self.hparams, "freeze_backbone_epochs", 0) or 0)
        if freeze_epochs > 0 and self._backbone_frozen and self.current_epoch >= freeze_epochs:
            try:
                self.model.unfreeze_backbone()
                self._backbone_frozen = False
                print(f"[freeze] Unfroze backbone at epoch {self.current_epoch}")
            except Exception as e:
                print(f"[freeze] Failed to unfreeze backbone: {e}")

    def _apply_mixup(self, x, y):
        """
        Apply MixUp augmentation if enabled.
        
        Returns:
            x: (Possibly mixed) images
            y_a: Original labels
            y_b: Mixed labels (same as y_a if no mixup)
            lam: Mixing coefficient (1.0 if no mixup)
            use_mixup: Whether mixup was applied
        """
        use_mixup_aug = getattr(self.hparams, "use_mixup", False)
        mixup_alpha = getattr(self.hparams, "mixup_alpha", 0.4)
        mixup_prob = getattr(self.hparams, "mixup_prob", 0.5)
        
        if use_mixup_aug and self.training and torch.rand(1).item() < mixup_prob:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
            return x, y_a, y_b, lam, True
        return x, y, y, 1.0, False

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y, object_mask = batch
        
        emb, probabilities, attn_map = self(x, object_mask=object_mask)

        # Compute losses
        arc_logits_margin = self.model.arcface_head(emb, labels=y)
        
        # For CombinedLossV2, we need different handling
        if hasattr(self.main_loss_fn, 'classification_loss'):
            loss_ce = self.main_loss_fn.classification_loss(arc_logits_margin, y)
        else:
            loss_ce = self.main_loss_fn.arcface_criterion(arc_logits_margin, y)
        
        # Metric loss
        if hasattr(self.main_loss_fn, 'miner') and self.main_loss_fn.miner is not None:
            hard_pairs = self.main_loss_fn.miner(emb, y)
            loss_metric = self.main_loss_fn.metric_loss(emb, y, indices_tuple=hard_pairs)
        else:
            loss_metric = self.main_loss_fn.metric_loss(emb, y)
        
        arcface_weight = getattr(self.main_loss_fn, 'arcface_weight', 0.9)
        metric_weight = getattr(self.main_loss_fn, 'metric_weight', 0.1)
        loss_total = arcface_weight * loss_ce + metric_weight * loss_metric
        
        # Log
        self.log("val/loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_ce", loss_ce, on_step=False, on_epoch=True)
        self.log("val/loss_metric", loss_metric, on_step=False, on_epoch=True)
        self.log("val/loss_total", loss_total, on_step=False, on_epoch=True)

        # Update accuracy
        preds = torch.argmax(probabilities, dim=1)
        self.val_accuracy.update(preds, y)
        self.val_accuracy_macro.update(preds, y)
        self.val_accuracy_top5.update(probabilities, y)
        self.val_accuracy_top10.update(probabilities, y)

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        self.log("val/accuracy_epoch", self.val_accuracy.compute(), prog_bar=True)
        self.val_accuracy.reset() 
        self.log("val/accuracy_macro_epoch", self.val_accuracy_macro.compute(), prog_bar=False)
        self.val_accuracy_macro.reset()
        self.log("val/accuracy_top5_epoch", self.val_accuracy_top5.compute(), prog_bar=False)
        self.val_accuracy_top5.reset()
        self.log("val/accuracy_top10_epoch", self.val_accuracy_top10.compute(), prog_bar=False)
        self.val_accuracy_top10.reset()

    def on_validation_epoch_start(self):
        """Visualize attention maps at the start of validation."""
        if not getattr(self.hparams, 'visualize_attention_map', False):
            return
            
        dataset = self.trainer.datamodule.val_dataloader().dataset
        visualize_dir = os.path.join(self.hparams.output_dir, "visualize")
        os.makedirs(visualize_dir, exist_ok=True)

        for img_id in tqdm(self.visualization_indices, desc="Visualizing Attention"):
            tensor_img, _, object_mask = dataset[img_id]
            tensor_img_batch = tensor_img.unsqueeze(0).to(self.device)
            object_mask_batch = object_mask.unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, _, attn_map = self.model(tensor_img_batch, object_mask=object_mask_batch)

            if attn_map is None:
                continue
            attn_map = attn_map[0]

            grid = vutils.make_grid(tensor_img, normalize=True)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            pil_image = Image.fromarray(ndarr)

            img_subfolder = os.path.join(visualize_dir, f"img_{img_id}")
            os.makedirs(img_subfolder, exist_ok=True)
            img_path = os.path.join(img_subfolder, f"epoch_{self.current_epoch}.png")
            
            save_attention_overlay(pil_image, attn_map, save_path=img_path, title=f"Epoch: {self.current_epoch}")

            # Save object mask for debugging
            try:
                mask = object_mask
                if isinstance(mask, torch.Tensor):
                    m = mask.detach().float()
                    if m.ndim == 3:
                        m = m[0]
                    elif m.ndim == 2:
                        pass
                    else:
                        m = m.squeeze()
                    m = (m > 0).to(torch.uint8) * 255
                    mask_pil = Image.fromarray(m.cpu().numpy())
                    mask_pil.save(os.path.join(img_subfolder, f"epoch_{self.current_epoch}_mask.png"))
            except Exception as e:
                print(f"[visualize] Failed to save mask for img_{img_id}: {e}")


class ImageEmbeddingTrainerViT(BaseImageEmbeddingTrainer):
    """
    Trainer for ViT-based models.
    
    Supports all ViT variants from timm including:
    - BEiT, DeiT, ViT
    - MaxViT, MaxxViT
    - EVA, DINOv2
    - Swin Transformer
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        backbone_model_name: str = 'beitv2_base_patch16_224.in1k_ft_in22k_in1k',
        arcface_s: float = 64.0,
        arcface_m: float = 0.2,
        arcface_weight: float = 0.9,
        metric_weight: float = 0.1,
        label_smoothing: float = 0.1,
        freeze_backbone_epochs: int = 0,
        attention_warmup_epochs: int = 0,
        embedding_dropout_rate: Optional[float] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        lr_eta_min: float = 1e-7,
        attention_loss_lambda: float = 0.15,
        load_checkpoint: Optional[str] = None,
        output_dir: str = "output_dir",
        visualize_attention_map: bool = False,
        # New parameters
        loss_type: str = 'combined',
        metric_loss_type: str = 'threshold_consistent',
        miner_type: str = 'batch_hard',
        use_cross_batch_memory: bool = False,
        memory_size: int = 4096,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        focal_weight: float = 0.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        pooling_type: str = 'attention',
        # Cyclic LR parameters
        use_cyclic_lr: bool = False,
        cyclic_mode: str = 'warm_restarts',
        cyclic_t0: int = 10,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._vit_downsampler = None
        
        vit_kwargs = dict(
            embedding_dim=self.hparams.embedding_dim,
            num_classes=self.hparams.num_classes,
            backbone_model_name=self.hparams.backbone_model_name,
            arcface_s=self.hparams.arcface_s,
            arcface_m=self.hparams.arcface_m,
            pooling_type=self.hparams.pooling_type,
        )
        if self.hparams.embedding_dropout_rate is not None:
            vit_kwargs["embedding_dropout_rate"] = float(self.hparams.embedding_dropout_rate)
        
        self.model = StableEmbeddingModelViT(**vit_kwargs)
        
        if load_checkpoint:
            self._load_weights(load_checkpoint)
        
        self._setup_common_components(num_classes, embedding_dim)

    def training_step(self, batch, batch_idx):
        x, y, object_mask = batch
        
        # Apply MixUp if enabled
        x, y_a, y_b, lam, use_mixup = self._apply_mixup(x, y)
        
        # --- Attention Guidance Logic ---
        features = self.model.backbone_feature_extractor(x)
        tokens, grid = self.model._tokens_and_grid_from_features(features)
        raw_attention_scores = self.model.pooling.attention_net(tokens)  # [B, N, 1]

        # Prepare target mask for attention guidance
        object_mask_float = object_mask.float()
        if object_mask_float.ndim == 3:
            object_mask_float = object_mask_float.unsqueeze(1)

        target_for_loss = None
        if grid is not None:
            H, W = grid
            target_grid = F.interpolate(object_mask_float, size=(H, W), mode="nearest")
            target_for_loss = (target_grid.flatten(1) > 0).float()
        else:
            target_for_loss = torch.zeros(
                (tokens.shape[0], tokens.shape[1]), 
                device=tokens.device, 
                dtype=torch.float32
            )
        
        loss_attention_guidance = self.attention_guidance_loss_fn(
            raw_attention_scores.squeeze(-1), 
            target_for_loss
        )

        # --- Main Training Logic ---
        weights = F.softmax(raw_attention_scores, dim=1)
        pooled = (tokens * weights).sum(dim=1)
        emb_raw = self.model.embedding_fc(pooled)
        emb = F.normalize(emb_raw, p=2, dim=1)
        
        # Compute loss
        if use_mixup:
            arc_logits_a = self.model.arcface_head(emb, labels=y_a)
            arc_logits_b = self.model.arcface_head(emb, labels=y_b)
            
            if hasattr(self.main_loss_fn, 'classification_loss'):
                loss_main = lam * self.main_loss_fn(emb, arc_logits_a, y_a) + \
                           (1 - lam) * self.main_loss_fn(emb, arc_logits_b, y_b)
            else:
                loss_main = lam * self.main_loss_fn(emb, arc_logits_a, y_a) + \
                           (1 - lam) * self.main_loss_fn(emb, arc_logits_b, y_b)
            arc_logits = arc_logits_a  # For accuracy calculation
        else:
            arc_logits = self.model.arcface_head(emb, labels=y)
            loss_main = self.main_loss_fn(emb, arc_logits, y)

        total_loss = loss_main + self._get_attention_lambda() * loss_attention_guidance
        
        # Logging
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_main", loss_main, on_step=False, on_epoch=True)
        self.log("train/loss_attn_guidance", loss_attention_guidance, on_step=False, on_epoch=True)
        
        # Accuracy on raw logits
        with torch.no_grad():
            raw_logits = self.model.arcface_head(emb)
            preds = torch.argmax(raw_logits, dim=1)
            self.train_accuracy.update(preds, y_a if use_mixup else y)
            self.train_accuracy_top5.update(raw_logits, y_a if use_mixup else y)
            self.train_accuracy_top10.update(raw_logits, y_a if use_mixup else y)
        
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy_top5", self.train_accuracy_top5, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/accuracy_top10", self.train_accuracy_top10, on_step=True, on_epoch=True, prog_bar=False)
        
        return total_loss


class ImageEmbeddingTrainerConvnext(BaseImageEmbeddingTrainer):
    """
    Trainer for CNN-based models (ConvNeXt, EfficientNet, etc.).
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        backbone_model_name: str = 'convnext_tiny',
        arcface_s: float = 64.0,
        arcface_m: float = 0.2,
        arcface_weight: float = 0.9,
        metric_weight: float = 0.1,
        label_smoothing: float = 0.1,
        freeze_backbone_epochs: int = 0,
        attention_warmup_epochs: int = 0,
        embedding_dropout_rate: Optional[float] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        lr_eta_min: float = 1e-7,
        attention_loss_lambda: float = 0.15,
        load_checkpoint: Optional[str] = None,
        output_dir: str = "output_dir",
        visualize_attention_map: bool = False,
        custom_backbone=None,
        backbone_out_features: Optional[int] = None,
        # New parameters
        loss_type: str = 'combined',
        metric_loss_type: str = 'threshold_consistent',
        miner_type: str = 'batch_hard',
        use_cross_batch_memory: bool = False,
        memory_size: int = 4096,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        focal_weight: float = 0.0,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        pooling_type: str = 'attention',
        # Cyclic LR parameters
        use_cyclic_lr: bool = False,
        cyclic_mode: str = 'warm_restarts',
        cyclic_t0: int = 10,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        conv_kwargs = dict(
            embedding_dim=self.hparams.embedding_dim,
            num_classes=self.hparams.num_classes,
            backbone_model_name=self.hparams.backbone_model_name,
            arcface_s=self.hparams.arcface_s,
            arcface_m=self.hparams.arcface_m,
            custom_backbone=self.hparams.custom_backbone,
            backbone_out_features=self.hparams.backbone_out_features,
            pooling_type=self.hparams.pooling_type,
        )
        if self.hparams.embedding_dropout_rate is not None:
            conv_kwargs["embedding_dropout_rate"] = float(self.hparams.embedding_dropout_rate)
        
        self.model = StableEmbeddingModel(**conv_kwargs)
        
        if load_checkpoint:
            self._load_weights(load_checkpoint)

        self._setup_common_components(num_classes, embedding_dim)

    def training_step(self, batch, batch_idx):
        x, y, object_mask = batch
        
        # Apply MixUp if enabled
        x, y_a, y_b, lam, use_mixup = self._apply_mixup(x, y)
        
        # --- Attention Guidance Logic ---
        # Only compute attention guidance if using attention pooling
        if self.hparams.pooling_type == 'attention':
            with torch.no_grad():
                features = self.model.backbone_feature_extractor(x)
                
            raw_attention_scores = self.model.pooling.attention_conv(features)
            
            B, _, H_attn, W_attn = raw_attention_scores.shape
            
            object_mask_gt_for_loss = object_mask.float().to(self.device)
            if object_mask_gt_for_loss.ndim == 3:
                object_mask_gt_for_loss = object_mask_gt_for_loss.unsqueeze(1)

            target_attention_mask = F.interpolate(
                object_mask_gt_for_loss, 
                size=(H_attn, W_attn), 
                mode='nearest'
            )
            
            loss_attention_guidance = self.attention_guidance_loss_fn(
                raw_attention_scores, 
                target_attention_mask
            )
        else:
            loss_attention_guidance = torch.tensor(0.0, device=self.device)
        
        # --- Main Training Logic ---
        if use_mixup:
            emb, arc_logits_a, _ = self(x, labels=y_a, object_mask=object_mask)
            _, arc_logits_b, _ = self(x, labels=y_b, object_mask=object_mask)
            
            loss_main = lam * self.main_loss_fn(emb, arc_logits_a, y_a) + \
                       (1 - lam) * self.main_loss_fn(emb, arc_logits_b, y_b)
            arc_logits = arc_logits_a
        else:
            emb, arc_logits, _ = self(x, labels=y, object_mask=object_mask)
            loss_main = self.main_loss_fn(emb, arc_logits, y)

        total_loss = loss_main + self._get_attention_lambda() * loss_attention_guidance
        
        # Logging
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_main", loss_main, on_step=False, on_epoch=True)
        self.log("train/loss_attn_guidance", loss_attention_guidance, on_step=False, on_epoch=True)
        
        # Accuracy on raw logits
        with torch.no_grad():
            raw_logits = self.model.arcface_head(emb)
            preds = torch.argmax(raw_logits, dim=1)
            self.train_accuracy.update(preds, y_a if use_mixup else y)
            self.train_accuracy_top5.update(raw_logits, y_a if use_mixup else y)
            self.train_accuracy_top10.update(raw_logits, y_a if use_mixup else y)
            
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy_top5", self.train_accuracy_top5, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/accuracy_top10", self.train_accuracy_top10, on_step=True, on_epoch=True, prog_bar=False)
        
        return total_loss
