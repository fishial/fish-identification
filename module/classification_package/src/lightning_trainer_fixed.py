# -*- coding: utf-8 -*-
"""
PyTorch Lightning Trainers for Fish Classification.

This module provides Lightning modules for training image embedding models
with support for:
- Multiple backbone architectures (ViT, ConvNeXt, etc.)
- Multiple loss functions (Combined, CombinedV2 with Focal, etc.)
- Attention guidance for improved localization
- Configurable pooling strategies

Key Classes:
- BaseImageEmbeddingTrainer: Base class with shared functionality
- ImageEmbeddingTrainerViT: For Vision Transformer backbones
"""

import os
import time
import random
import logging
from typing import Optional, Literal, List, Tuple, Union

_module_logger = logging.getLogger(__name__)

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
from pytorch_metric_learning.losses import CrossBatchMemory

from module.classification_package.src.model_v2 import StableEmbeddingModelViT
from module.classification_package.src.loss_functions import (
    create_loss_function,
    compute_class_weights,
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
    - Fisher Ratio calculation for representation quality monitoring 🔥
    """
    
    # Box-drawing constants for summary output
    _BOX_TL = "╔"
    _BOX_TR = "╗"
    _BOX_BL = "╚"
    _BOX_BR = "╝"
    _BOX_H  = "═"
    _BOX_V  = "║"
    _BOX_ML = "╠"
    _BOX_MR = "╣"
    _BOX_SL = "╟"
    _BOX_SR = "╢"
    _BOX_SH = "─"
    _BOX_WIDTH = 74

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
        
        # Fisher Ratio storage
        self.val_embeddings = []
        self.val_labels = []
        # Fisher cap for val: set in on_validation_epoch_start; lazy fallback in validation_step for sanity_val ordering
        self._fisher_val_budget: Optional[int] = None
        self._fisher_val_budget_ready: bool = False

        # Timing & tracking
        self._fit_start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._epoch_train_steps: int = 0
        self._epoch_train_samples: int = 0

        # Best metric tracking
        self._best_val_acc: float = 0.0
        self._best_val_acc_epoch: int = -1
        self._best_fisher_ratio: float = 0.0
        self._best_fisher_epoch: int = -1
        self._epoch_history: List[dict] = []

    # ──────────────────────────────────────────────────────────────────
    # Formatting helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"

    @staticmethod
    def _format_num(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    @staticmethod
    def _format_metric(v: float, decimals: int = 4) -> str:
        if abs(v) < 1e-3 or abs(v) > 1e4:
            return f"{v:.3e}"
        return f"{v:.{decimals}f}"

    def _get_current_lr(self) -> float:
        if self.trainer and self.trainer.optimizers:
            return self.trainer.optimizers[0].param_groups[0]["lr"]
        return 0.0

    def _get_gpu_memory(self) -> Tuple[Optional[float], Optional[float]]:
        """Returns (current_GB, peak_GB) or (None, None) if no CUDA."""
        if not torch.cuda.is_available():
            return None, None
        try:
            dev = self.device
            current = torch.cuda.memory_allocated(dev) / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
            return current, peak
        except Exception:
            return None, None

    def _count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def _box_line(self, text: str = "", style: str = "content") -> str:
        w = self._BOX_WIDTH
        inner = w - 2
        if style == "top":
            return f"{self._BOX_TL}{self._BOX_H * inner}{self._BOX_TR}"
        if style == "bottom":
            return f"{self._BOX_BL}{self._BOX_H * inner}{self._BOX_BR}"
        if style == "mid":
            return f"{self._BOX_ML}{self._BOX_H * inner}{self._BOX_MR}"
        if style == "sep":
            return f"{self._BOX_SL}{self._BOX_SH * inner}{self._BOX_SR}"
        padded = f" {text}".ljust(inner)[:inner]
        return f"{self._BOX_V}{padded}{self._BOX_V}"

    def _print_box(self, title: str, sections: List[Union[str, Tuple, None]]):
        """Print a formatted summary box.

        ``sections`` items:
        - ``None``               -> separator line
        - ``str``                -> full-width content line
        - ``(key, value)``       -> single key-value row
        - ``(k1, v1, k2, v2)``  -> two-column key-value row
        """
        lines = [self._box_line(style="top"), self._box_line(f"  {title}  ".center(self._BOX_WIDTH - 2))]
        lines.append(self._box_line(style="mid"))

        for item in sections:
            if item is None:
                lines.append(self._box_line(style="sep"))
            elif isinstance(item, str):
                lines.append(self._box_line(item))
            elif isinstance(item, tuple):
                if len(item) == 2:
                    k, v = item
                    v_str = self._format_metric(v) if isinstance(v, float) else str(v)
                    lines.append(self._box_line(f"  {k:<32s} {v_str}"))
                elif len(item) == 4:
                    k1, v1, k2, v2 = item
                    fv = lambda v: self._format_metric(v) if isinstance(v, float) else str(v)
                    half = (self._BOX_WIDTH - 6) // 2
                    left = f"{k1}: {fv(v1)}"
                    right = f"{k2}: {fv(v2)}"
                    lines.append(self._box_line(f"  {left:<{half}s}{right}"))

        lines.append(self._box_line(style="bottom"))
        print("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────

    def _get_attention_lambda(self) -> float:
        """Get attention loss weight, considering warmup."""
        warmup = int(getattr(self.hparams, "attention_warmup_epochs", 0) or 0)
        if self.current_epoch < warmup:
            return 0.0
        return float(getattr(self.hparams, "attention_loss_lambda", 0.0) or 0.0)

    def _setup_common_components(self, num_classes: int, embedding_dim: int = 512, class_weights=None):
        """
        Initialize losses and metrics common to all trainers.
        """
        loss_type = getattr(self.hparams, "loss_type", "combined")
        metric_loss_type = getattr(self.hparams, "metric_loss_type", "threshold_consistent")
        miner_type = getattr(self.hparams, "miner_type", "batch_hard")
        use_cross_batch_memory = getattr(self.hparams, "use_cross_batch_memory", False)
        memory_size = getattr(self.hparams, "memory_size", 4096)
        focal_gamma = getattr(self.hparams, "focal_gamma", 2.0)
        focal_alpha = getattr(self.hparams, "focal_alpha", 0.25)
        focal_weight = getattr(self.hparams, "focal_weight", 0.0)
        
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
            class_weights=class_weights,
        )
        
        # Attention guidance loss (pos_weight is computed dynamically in training_step)
        self.attention_guidance_loss_fn = None

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
        # ... (Code unchanged; leave as is) ...
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
        return self.model(
            x, 
            labels=labels, 
            object_mask=object_mask, 
            return_softmax=(not self.training),
            return_attention_map=True
        )

    def configure_optimizers(self):
        from timm.scheduler import CosineLRScheduler
        from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingWarmRestarts

        no_decay_keywords = {'bias', 'bn', 'norm', 'cls_gate', 'temperature'}
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name.lower() for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW([
            {'params': decay_params, 'weight_decay': self.hparams.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.hparams.lr)
        
        use_cyclic_lr = getattr(self.hparams, "use_cyclic_lr", False)
        
        if use_cyclic_lr:
            cyclic_mode = getattr(self.hparams, "cyclic_mode", "warm_restarts")
            if cyclic_mode == "warm_restarts":
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
                max_epochs = getattr(self.hparams, "max_epochs", self.trainer.max_epochs)
                steps_per_epoch = 100 
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
            # Smart calculation: when target LR is below 1e-6, warmup starts at that low value as well
            smart_warmup_lr = min(self.hparams.lr, 1e-6)
            # When debugging (few epochs), disable warmup; otherwise keep 5 epochs
            smart_warmup_t = 5 if self.trainer.max_epochs > 10 else 0

            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.trainer.max_epochs,
                lr_min=self.hparams.lr_eta_min,
                warmup_t=smart_warmup_t,
                warmup_lr_init=smart_warmup_lr,
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
        # ... (Code unchanged) ...
        from timm.scheduler import CosineLRScheduler
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
        
        if isinstance(scheduler, CosineLRScheduler):
            scheduler.step(epoch=self.current_epoch)
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, CyclicLR):
            pass 
        else:
            if hasattr(scheduler, 'step'):
                scheduler.step()

    def on_fit_start(self):
        self._fit_start_time = time.time()
        seed = 42
        pl.seed_everything(seed, workers=True)

        train_segmentation_head = getattr(self.hparams, "train_segmentation_head", False)
        if train_segmentation_head:
            self.model.freeze_everything_except_segmentation_head()
            self._backbone_frozen = True # so the script knows the backbone is frozen

        count_random_img = 63
        self.visualization_indices = set()
        if hasattr(self.trainer.datamodule, "val_dataloader"):
            val_loader = self.trainer.datamodule.val_dataloader()
            if val_loader is not None and getattr(val_loader, "dataset", None) is not None:
                total_val_size = len(val_loader.dataset)
                count_to_sample = min(count_random_img, total_val_size)
                if count_to_sample > 0:
                    self.visualization_indices = set(random.sample(range(total_val_size), count_to_sample))

        freeze_epochs = int(getattr(self.hparams, "freeze_backbone_epochs", 0) or 0)
        if freeze_epochs > 0 and not train_segmentation_head:
            try:
                self.model.freeze_backbone()
                self._backbone_frozen = True
            except Exception as e:
                print(f"[freeze] Failed to freeze backbone: {e}")

        # ── Print training configuration summary ──
        params = self._count_parameters()
        backbone = getattr(self.hparams, "backbone_model_name", "unknown")
        num_classes = getattr(self.hparams, "num_classes", "?")
        emb_dim = getattr(self.hparams, "embedding_dim", "?")
        loss_type = getattr(self.hparams, "loss_type", "combined")
        pooling = getattr(self.hparams, "pooling_type", "?")
        neck = getattr(self.hparams, "neck_type", "simple")
        head = getattr(self.hparams, "head_type", "arcface")
        img_size = getattr(self.hparams, "input_img_size", "?")

        train_size = val_size = "?"
        if hasattr(self.trainer.datamodule, "train_dataloader"):
            try:
                tl = self.trainer.datamodule.train_dataloader()
                train_size = self._format_num(len(tl.dataset))
            except Exception:
                pass
        if hasattr(self.trainer.datamodule, "val_dataloader"):
            try:
                vl = self.trainer.datamodule.val_dataloader()
                val_size = self._format_num(len(vl.dataset))
            except Exception:
                pass

        sections = [
            ("Backbone", str(backbone)),
            ("Image size", str(img_size)),
            ("Pooling / Neck / Head", f"{pooling} / {neck} / {head}"),
            ("Classes", str(num_classes)),
            ("Embedding dim", str(emb_dim)),
            None,
            ("Total parameters", self._format_num(params["total"])),
            ("Trainable parameters", self._format_num(params["trainable"])),
            ("Frozen parameters", self._format_num(params["frozen"])),
            None,
            ("Loss type", str(loss_type)),
            ("ArcFace weight", getattr(self.hparams, "arcface_weight", "?")),
            ("Metric weight", getattr(self.hparams, "metric_weight", "?")),
            ("Label smoothing", getattr(self.hparams, "label_smoothing", "?")),
            ("Attention lambda", getattr(self.hparams, "attention_loss_lambda", 0.0)),
            None,
            ("Learning rate", getattr(self.hparams, "lr", "?")),
            ("Weight decay", getattr(self.hparams, "weight_decay", "?")),
            ("LR min", getattr(self.hparams, "lr_eta_min", "?")),
            ("Freeze backbone epochs", str(freeze_epochs)),
            None,
            ("Train samples", str(train_size)),
            ("Val samples", str(val_size)),
            ("Max epochs", str(self.trainer.max_epochs)),
            ("Precision", str(self.trainer.precision)),
        ]

        self._print_box("Training Configuration", sections)

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()
        self._epoch_train_steps = 0
        self._epoch_train_samples = 0

        max_ep = self.trainer.max_epochs or "?"
        lr = self._get_current_lr()
        frozen_tag = " [backbone frozen]" if self._backbone_frozen else ""
        print(f"\n{'─' * 74}")
        print(f"  Epoch {self.current_epoch}/{max_ep}{frozen_tag}  |  LR: {lr:.2e}")
        print(f"{'─' * 74}")

        freeze_epochs = int(getattr(self.hparams, "freeze_backbone_epochs", 0) or 0)
        train_segmentation_head = getattr(self.hparams, "train_segmentation_head", False)

        if (
            freeze_epochs > 0
            and self._backbone_frozen
            and self.current_epoch >= freeze_epochs
            and not train_segmentation_head
        ):
            try:
                self.model.unfreeze_backbone()
                self._backbone_frozen = False
                params = self._count_parameters()
                print(f"  >> Backbone unfrozen at epoch {self.current_epoch} "
                      f"({self._format_num(params['trainable'])} trainable params)")
            except Exception as e:
                print(f"  >> Failed to unfreeze backbone: {e}")

        use_dynamic_margin = getattr(self.hparams, "use_dynamic_margin", False)
        if use_dynamic_margin and hasattr(self.model, "arcface_head"):
            m_start = getattr(self.hparams, "arcface_m_start", 0.1)
            m_end = float(self.hparams.arcface_m)
            max_epochs = self.trainer.max_epochs or 100
            progress = min(self.current_epoch / (max_epochs * 0.7), 1.0)
            new_m = m_start + (m_end - m_start) * progress
            self.model.arcface_head.set_margin(new_m)
            self.log("train/arcface_margin", new_m, on_step=False, on_epoch=True)

        head = getattr(self.model, "arcface_head", None)
        if head is not None and getattr(head, "use_adacos", False):
            self.log("train/adacos_scale", head.s.item(), on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        throughput = self._epoch_train_samples / elapsed if elapsed > 0 else 0

        self.log("train/epoch_time_sec", elapsed, on_step=False, on_epoch=True)
        self.log("train/throughput_img_sec", throughput, on_step=False, on_epoch=True)

        gpu_cur, gpu_peak = self._get_gpu_memory()
        if gpu_cur is not None:
            self.log("train/gpu_mem_gb", gpu_cur, on_step=False, on_epoch=True)
            self.log("train/gpu_mem_peak_gb", gpu_peak, on_step=False, on_epoch=True)

    def _init_fisher_val_budget(self) -> None:
        cap = getattr(self.hparams, "max_fisher_val_samples", 8192)
        self._fisher_val_budget = None if cap is None or cap < 0 else int(cap)
        self._fisher_val_budget_ready = True

    def on_validation_epoch_start(self):
        self.val_embeddings.clear()
        self.val_labels.clear()
        self._init_fisher_val_budget()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if not self._fisher_val_budget_ready:
            self.val_embeddings.clear()
            self.val_labels.clear()
            self._init_fisher_val_budget()

        x, y, object_mask = batch

        with torch.inference_mode():
            out = self.model.forward_train(x, labels=None, object_mask=object_mask)
            emb = out["emb"]

            clean_logits = self.model.arcface_head(emb, labels=None)
            probabilities = F.softmax(clean_logits, dim=1)
            preds = torch.argmax(probabilities, dim=1)

            arc_logits_margin = self.model.arcface_head(emb, labels=y)

            if hasattr(self.main_loss_fn, 'classification_loss'):
                loss_ce = self.main_loss_fn.classification_loss(arc_logits_margin, y)
            else:
                loss_ce = self.main_loss_fn.arcface_criterion(arc_logits_margin, y)

            metric_loss_fn = self.main_loss_fn.metric_loss
            if isinstance(metric_loss_fn, CrossBatchMemory):
                metric_loss_fn = metric_loss_fn.loss

            if hasattr(self.main_loss_fn, 'miner') and self.main_loss_fn.miner is not None:
                hard_pairs = self.main_loss_fn.miner(emb, y)
                loss_metric = metric_loss_fn(emb, y, indices_tuple=hard_pairs)
            else:
                loss_metric = metric_loss_fn(emb, y)

            arcface_weight = getattr(self.main_loss_fn, 'arcface_weight', 0.9)
            metric_weight = getattr(self.main_loss_fn, 'metric_weight', 0.1)

            # === NEW BLOCK: Segmentation (Loss, Metrics, and Visualization) ===
            loss_segmentation = torch.tensor(0.0, device=self.device)
            if "seg_logits" in out and out["seg_logits"] is not None and object_mask is not None:
                # 1. Resize
                pred_mask_logits = F.interpolate(
                    out["seg_logits"], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
                )

                # 2. Prepare Ground Truth
                gt_mask = object_mask.float()
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask.unsqueeze(1)  # [B, 1, H, W]

                # 3. Validation BCE Loss
                loss_segmentation = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask)
                self.log("val/loss_seg", loss_segmentation, on_step=False, on_epoch=True, sync_dist=True)

                # 4. IoU metric
                preds_bin = (pred_mask_logits > 0.0).float()
                intersection = (preds_bin * gt_mask).sum(dim=(1, 2, 3))
                union = preds_bin.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                self.log("val/iou_seg", iou.mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

                # 5. TensorBoard visualization (only for the first batch of the epoch)
                if batch_idx == 0:
                    n_vis = min(4, x.shape[0])
                    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                    x_vis = (x[:n_vis] * std + mean).clamp(0, 1)

                    true_m = gt_mask[:n_vis].repeat(1, 3, 1, 1)
                    pred_m = torch.sigmoid(pred_mask_logits[:n_vis]).repeat(1, 3, 1, 1)

                    grid = vutils.make_grid(torch.cat([x_vis, true_m, pred_m], dim=0), nrow=n_vis)

                    if hasattr(self.logger, 'experiment'):
                        try:
                            self.logger.experiment.add_image(
                                "val/Segmentation_True_vs_Pred", grid, self.current_epoch
                            )
                        except Exception:
                            pass
            # ================================================================

            # Update the total validation loss
            seg_lambda = getattr(self.hparams, "segmentation_loss_lambda", 1.0)

            loss_total = (
                arcface_weight * loss_ce
                + metric_weight * loss_metric
                + seg_lambda * loss_segmentation
            )

            # Log
            self.log("val/loss", loss_total, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/loss_ce", loss_ce, on_step=False, on_epoch=True)
            self.log("val/loss_metric", loss_metric, on_step=False, on_epoch=True)
            self.log("val/loss_total", loss_total, on_step=False, on_epoch=True)

            self.val_accuracy.update(preds, y)
            self.val_accuracy_macro.update(preds, y)
            self.val_accuracy_top5.update(probabilities, y)
            self.val_accuracy_top10.update(probabilities, y)

            # Fisher ratio: cap stored CPU tensors to limit RAM (full val set ~30k × dim float32 adds up with DataLoader buffers)
            budget = self._fisher_val_budget
            if budget is None:
                self.val_embeddings.append(emb.detach().cpu())
                self.val_labels.append(y.detach().cpu())
            elif budget > 0:
                take = min(emb.shape[0], budget)
                if take > 0:
                    self.val_embeddings.append(emb[:take].detach().cpu())
                    self.val_labels.append(y[:take].detach().cpu())
                    self._fisher_val_budget = budget - take

    def on_validation_epoch_end(self):
        """Compute metrics, log them, and print a consolidated epoch summary."""

        # ── Safeguard: initialize best-value trackers if they don't exist yet ──
        if not hasattr(self, '_best_val_loss'): self._best_val_loss = float('inf')
        if not hasattr(self, '_best_train_loss'): self._best_train_loss = float('inf')
        if not hasattr(self, '_best_val_iou_seg'): self._best_val_iou_seg = 0.0

        # ── Compute & log accuracy metrics ──
        acc = self.val_accuracy.compute()
        acc_macro = self.val_accuracy_macro.compute()
        acc_top5 = self.val_accuracy_top5.compute()
        acc_top10 = self.val_accuracy_top10.compute()

        self.log("val/accuracy_epoch", acc, prog_bar=False)
        self.log("val_acc_int", (acc * 10000).int(), prog_bar=False)
        self.log("val/accuracy_macro_epoch", acc_macro, prog_bar=True)
        self.log("val/accuracy_top5_epoch", acc_top5, prog_bar=False)
        self.log("val/accuracy_top10_epoch", acc_top10, prog_bar=False)

        self.val_accuracy.reset()
        self.val_accuracy_macro.reset()
        self.val_accuracy_top5.reset()
        self.val_accuracy_top10.reset()

        acc_val = acc.item()
        acc_macro_val = acc_macro.item()
        acc_top5_val = acc_top5.item()
        acc_top10_val = acc_top10.item()

        # ── Fisher Ratio ──
        fisher_ratio_val = None
        mean_intra_val = None
        mean_inter_val = None

        if len(self.val_embeddings) > 0:
            embs = torch.cat(self.val_embeddings)
            lbls = torch.cat(self.val_labels)

            unique_labels = torch.unique(lbls)
            centroids = []
            intra_distances = []

            for lbl in unique_labels:
                mask = lbls == lbl
                class_embs = embs[mask]
                if len(class_embs) < 2:
                    continue
                centroid = F.normalize(class_embs.mean(dim=0), p=2, dim=0)
                centroids.append(centroid)
                sims = F.cosine_similarity(class_embs, centroid.unsqueeze(0))
                intra_distances.append((1.0 - sims).mean())

            if len(centroids) > 1 and len(intra_distances) > 0:
                centroids_tensor = torch.stack(centroids)
                sim_matrix = torch.mm(centroids_tensor, centroids_tensor.t())
                dist_matrix = 1.0 - sim_matrix
                triu_idx = torch.triu_indices(len(centroids), len(centroids), offset=1)
                inter_distances = dist_matrix[triu_idx[0], triu_idx[1]]

                mean_intra = torch.stack(intra_distances).mean()
                mean_inter = inter_distances.mean()
                fisher_ratio = mean_inter / (mean_intra + 1e-10)

                self.log("val/fisher_ratio", fisher_ratio, sync_dist=True, prog_bar=True)
                self.log("val/intra_var", mean_intra, sync_dist=True)
                self.log("val/inter_dist", mean_inter, sync_dist=True)

                fisher_ratio_val = fisher_ratio.item()
                mean_intra_val = mean_intra.item()
                mean_inter_val = mean_inter.item()

            self.val_embeddings.clear()
            self.val_labels.clear()

        # ── Collect metrics from callback_metrics ──
        cb = self.trainer.callback_metrics
        train_loss = cb.get("train/loss_epoch")
        train_acc = cb.get("train/accuracy_epoch")
        val_loss = cb.get("val/loss")
        val_iou_seg = cb.get("val/iou_seg")
        val_loss_seg = cb.get("val/loss_seg")
        train_loss_seg = cb.get("train/loss_seg")

        # ── STORE PREVIOUS BEST VALUES (for delta calculation) ──
        old_best_acc = getattr(self, '_best_val_acc', 0.0)
        old_best_fisher = getattr(self, '_best_fisher_ratio', 0.0)
        old_best_val_loss = self._best_val_loss
        old_best_train_loss = self._best_train_loss
        old_best_val_iou = self._best_val_iou_seg

        # ── UPDATE THE CURRENT BEST VALUES ──
        is_new_best_acc = False
        if acc_val > self._best_val_acc:
            self._best_val_acc = acc_val
            self._best_val_acc_epoch = self.current_epoch
            is_new_best_acc = True

        if fisher_ratio_val is not None and fisher_ratio_val > self._best_fisher_ratio:
            self._best_fisher_ratio = fisher_ratio_val
            self._best_fisher_epoch = self.current_epoch
            
        if val_loss is not None and val_loss.item() < self._best_val_loss:
            self._best_val_loss = val_loss.item()
            
        if train_loss is not None and train_loss.item() < self._best_train_loss:
            self._best_train_loss = train_loss.item()
            
        if val_iou_seg is not None and val_iou_seg.item() > self._best_val_iou_seg:
            self._best_val_iou_seg = val_iou_seg.item()

        self.log("val/best_accuracy", self._best_val_acc, prog_bar=False)
        self.log("val/best_accuracy_epoch", float(getattr(self, '_best_val_acc_epoch', 0)), prog_bar=False)

        # ── SMART FORMATTER WITH DELTA ──
        def format_metric(current, old_best, is_loss=False, is_percent=False):
            """Format the value, append delta relative to the previous best, arrows, and a record tag."""
            val_mult = 100.0 if is_percent else 1.0
            unit = "%" if is_percent else ""
            
            base_str = f"{current * val_mult:.4f}{unit}"
            
            # If this is the first epoch or the previous best is infinity/zero
            if old_best is None or old_best == float('inf') or old_best == 0.0:
                return base_str

            delta = current - old_best
            d_val = abs(delta * val_mult)
            
            if d_val < 1e-5:
                return f"{base_str} (=)"

            sign = "+" if delta > 0 else "-"
            arrow = "↓" if delta < 0 else "↑"
            
            # Logic: loss should decrease (delta < 0), metrics should increase (delta > 0)
            is_new_best = (delta < 0) if is_loss else (delta > 0)
            best_tag = " ★ BEST" if is_new_best else ""
            
            return f"{base_str} ({sign}{d_val:.4f}{unit} {arrow}){best_tag}"

        # ── Timing ──
        epoch_time = time.time() - self._epoch_start_time if getattr(self, '_epoch_start_time', None) else 0
        total_time = time.time() - self._fit_start_time if getattr(self, '_fit_start_time', None) else 0
        remaining_epochs = (self.trainer.max_epochs or 0) - self.current_epoch - 1
        eta = epoch_time * remaining_epochs if remaining_epochs > 0 else 0

        # ── Build summary ──
        max_ep = self.trainer.max_epochs or "?"
        star = " 🌟 NEW OVERALL BEST 🌟" if is_new_best_acc else ""
        title = f"Epoch {self.current_epoch}/{max_ep} Summary{star}"

        sections = []

        # Training section
        sections.append(" TRAIN")
        if train_loss is not None:
            sections.append(("  Loss", format_metric(train_loss.item(), old_best_train_loss, is_loss=True)))
        if train_loss_seg is not None:
            sections.append(("  Loss Seg", f"{train_loss_seg.item():.4f}")) # Small losses can omit the delta
        if train_acc is not None:
            sections.append(("  Accuracy", f"{train_acc.item() * 100:.2f}%"))
        sections.append(("  Learning rate", self._get_current_lr()))
        sections.append(None)

        # Validation section
        sections.append(" VALIDATION")
        if val_loss is not None:
            sections.append(("  Loss", format_metric(val_loss.item(), old_best_val_loss, is_loss=True)))
        if val_loss_seg is not None:
            sections.append(("  Loss Seg", f"{val_loss_seg.item():.4f}"))
            
        sections.append(("  Accuracy (micro)", format_metric(acc_val, old_best_acc, is_percent=True)))
        sections.append(("  Accuracy (macro)", f"{acc_macro_val * 100:.2f}%"))
        sections.append(("  Accuracy Top-5", f"{acc_top5_val * 100:.2f}%"))
        
        if val_iou_seg is not None:
            sections.append(("  Segmentation IoU", format_metric(val_iou_seg.item(), old_best_val_iou, is_percent=True)))
            
        sections.append(("  Best accuracy", f"{self._best_val_acc * 100:.2f}% (epoch {getattr(self, '_best_val_acc_epoch', 0)})"))

        if fisher_ratio_val is not None:
            sections.append(None)
            sections.append(" EMBEDDING QUALITY")
            sections.append(("  Fisher ratio", format_metric(fisher_ratio_val, old_best_fisher)))
            sections.append(("  Intra-class dist", f"{mean_intra_val:.4f}"))
            sections.append(("  Inter-class dist", f"{mean_inter_val:.4f}"))
            if getattr(self, '_best_fisher_ratio', 0) > 0:
                sections.append(("  Best Fisher ratio", f"{self._best_fisher_ratio:.4f} (epoch {getattr(self, '_best_fisher_epoch', 0)})"))

        sections.append(None)
        sections.append(" RESOURCES")
        sections.append(("  Epoch time", self._format_time(epoch_time)))
        sections.append(("  Total elapsed", self._format_time(total_time)))
        if eta > 0:
            sections.append(("  ETA", self._format_time(eta)))
        throughput = getattr(self, '_epoch_train_samples', 0) / epoch_time if epoch_time > 0 else 0
        if throughput > 0:
            sections.append(("  Throughput", f"{throughput:.0f} img/s"))
        
        gpu_cur, gpu_peak = self._get_gpu_memory()
        if gpu_cur is not None:
            sections.append(("  GPU memory", f"{gpu_cur:.1f} GB (peak: {gpu_peak:.1f} GB)"))

        self._print_box(title, sections)

        # Store epoch history
        self._epoch_history.append({
            "epoch": self.current_epoch,
            "val_acc": acc_val,
            "val_acc_macro": acc_macro_val,
            "train_loss": train_loss.item() if train_loss is not None else None,
            "val_loss": val_loss.item() if val_loss is not None else None,
            "val_iou_seg": val_iou_seg.item() if val_iou_seg is not None else None,
            "fisher": fisher_ratio_val,
            "lr": self._get_current_lr(),
            "time": epoch_time,
        })

    def on_fit_end(self):
        total_time = time.time() - self._fit_start_time if self._fit_start_time else 0

        sections: List[Union[str, Tuple, None]] = [
            ("Total training time", self._format_time(total_time)),
            ("Epochs completed", str(self.current_epoch + 1)),
            None,
            ("Best val accuracy", f"{self._best_val_acc:.4f} (epoch {self._best_val_acc_epoch})"),
        ]
        if self._best_fisher_ratio > 0:
            sections.append(("Best Fisher ratio", f"{self._best_fisher_ratio:.4f} (epoch {self._best_fisher_epoch})"))

        if self._epoch_history:
            sections.append(None)
            sections.append(" METRIC TREND (last 5 epochs)")
            tail = self._epoch_history[-5:]
            for rec in tail:
                vacc = f"acc={rec['val_acc']:.4f}" if rec["val_acc"] is not None else ""
                vloss = f"loss={rec['val_loss']:.4f}" if rec["val_loss"] is not None else ""
                fr = f"fisher={rec['fisher']:.2f}" if rec.get("fisher") is not None else ""
                parts = [p for p in [vacc, vloss, fr] if p]
                sections.append(f"  ep {rec['epoch']:>3d}:  {' | '.join(parts)}")

        gpu_cur, gpu_peak = self._get_gpu_memory()
        if gpu_peak is not None:
            sections.append(None)
            sections.append(("Peak GPU memory", f"{gpu_peak:.1f} GB"))

        self._print_box("Training Complete", sections)

    def on_validation_epoch_start(self):
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
        coverage_loss_lambda: float = 0.1,
        diversity_loss_lambda: float = 0.1,
        segmentation_loss_lambda: float = 1.0,

        coverage_temperature: float = 3.0,  # FIX 5: was hardcoded 3.0 inside training_step
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
        pooling_type: str = 'attention',
        num_attention_heads: int = 8,
        use_cls_token: bool = True,
        # SubCenter ArcFace / AdaCos
        arcface_K: int = 3,
        use_adacos: bool = False,
        # Dynamic ArcFace margin
        use_dynamic_margin: bool = False,
        arcface_m_start: float = 0.1,
        # Cyclic LR parameters
        use_cyclic_lr: bool = False,
        cyclic_mode: str = 'warm_restarts',
        cyclic_t0: int = 10,
        max_epochs: int = 100,
        neck_type: str = 'simple',
        head_type: str = 'arcface',
        input_img_size: Optional[Tuple[int, int]] = None,

        # Segmentation-specific parameters
        train_segmentation_head: bool = False,

        # Backbone regularization
        drop_path_rate: float = 0.0,

        # Per-class weights for focal loss (computed externally)
        class_weights: Optional[torch.Tensor] = None,
        max_fisher_val_samples: int = 8192,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            _module_logger.debug(
                "Ignoring removed hyperparameters from checkpoint/config: %s",
                sorted(kwargs.keys()),
            )
        self.save_hyperparameters(ignore=["class_weights", "kwargs"])
        self._vit_downsampler = None
        self._class_weights = class_weights
        
        vit_kwargs = dict(
            embedding_dim=self.hparams.embedding_dim,
            num_classes=self.hparams.num_classes,
            arcface_s=self.hparams.arcface_s,
            arcface_m=self.hparams.arcface_m,
            arcface_K=self.hparams.arcface_K,
            use_adacos=self.hparams.use_adacos,
            pooling_type=self.hparams.pooling_type,
            num_attention_heads=self.hparams.num_attention_heads,
            use_cls_token=self.hparams.use_cls_token,
            neck_type=self.hparams.neck_type,
            head_type=self.hparams.head_type,
            input_img_size=self.hparams.input_img_size,
            train_segmentation_head=self.hparams.train_segmentation_head,
            drop_path_rate=self.hparams.drop_path_rate,
        )
        if self.hparams.embedding_dropout_rate is not None:
            vit_kwargs["embedding_dropout_rate"] = float(self.hparams.embedding_dropout_rate)
        
        if load_checkpoint:
            print(f"  Loading checkpoint from {load_checkpoint}")
            self.model = StableEmbeddingModelViT.load_from_checkpoint(
                checkpoint_path=load_checkpoint,
                map_location=self.device,
                strict=False
            )
        else:
            print("No checkpoint provided. Initializing model with random weights.")
            self.model = StableEmbeddingModelViT(**vit_kwargs)
        
        if self.hparams.train_segmentation_head:
            self.model.freeze_everything_except_segmentation_head()

        self._setup_common_components(num_classes, embedding_dim, class_weights=self._class_weights)

    def training_step(self, batch, batch_idx):
        x, y, object_mask = batch
        current_batch_size = x.size(0)

        self._epoch_train_steps += 1
        self._epoch_train_samples += x.shape[0]

        out = self.model.forward_train(x, labels=y, object_mask=object_mask)
        emb = out["emb"]
        tokens = out["tokens"]
        grid = out["grid"]
        raw_attention_scores = out["raw_attention_scores"]

        # --- 1. Attention Losses (Guidance, Diversity, Coverage) ---
        loss_attention_guidance = torch.tensor(0.0, device=self.device)
        loss_diversity = torch.tensor(0.0, device=self.device)
        loss_coverage = torch.tensor(0.0, device=self.device)

        if raw_attention_scores is not None:
            object_mask_float = object_mask.float()
            if object_mask_float.ndim == 3:
                object_mask_float = object_mask_float.unsqueeze(1)

            if grid is not None:
                H, W = grid
                # Bilinear interpolation gives us patch coverage ratio (0.0 to 1.0)
                target_grid = F.interpolate(object_mask_float, size=(H, W), mode="bilinear", align_corners=False)
                
                # A 0.2 threshold preserves thin details (fins, tails) occupying at least 20% of a DINOv2 patch
                target_for_loss = (target_grid.flatten(1) > 0.2).float() # [B, N]
            else:
                target_for_loss = torch.zeros(
                    (tokens.shape[0], tokens.shape[1]),
                    device=tokens.device, dtype=torch.float32,
                )
            
            # === LOGIT PREPARATION (shared across losses) ===
            attn_logits = raw_attention_scores.squeeze(-1)
            if attn_logits.ndim == 3:
                attn_logits = attn_logits.transpose(1, 2)  # → [B, H, N]
                num_heads = attn_logits.shape[1]
            else:
                num_heads = 1

            attn_temperature = getattr(self.model.pooling, 'temperature', 1.0)

            # FIX 4: Compute attn_probs ONCE
            attn_probs = F.softmax(attn_logits / attn_temperature, dim=-1)

            # --- 1A. Guidance Loss (Background Leakage Penalty) ---
            background_mask = (1.0 - target_for_loss)
            if num_heads > 1:
                background_mask = background_mask.unsqueeze(1)

            leakage = (attn_probs * background_mask).sum(dim=-1)
            loss_attention_guidance = leakage.mean()

            # --- 1B. Masked Diversity Loss (Head orthogonality ONLY over the fish) ---
            if num_heads > 1:
                eps = 1e-6
                # Keep only the attention that landed on the object
                attn_masked = attn_probs * target_for_loss.unsqueeze(1) # [B, H, N]
                
                # Normalize the background-cleaned vectors
                attn_norm = attn_masked / (attn_masked.norm(dim=-1, keepdim=True) + eps)
                
                # Compute the cosine similarity matrix [B, H, H]
                sim_matrix = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
                
                # Build the ideal matrix (ones on the diagonal, zeros elsewhere)
                identity = torch.eye(num_heads, device=self.device).unsqueeze(0).expand(attn_probs.shape[0], -1, -1)
                
                loss_diversity = F.mse_loss(sim_matrix, identity)

            # --- 1C. Coverage Loss (Union Attention + Distribution Matching) ---
            if num_heads > 1:
                # FIX 5: coverage_temperature is now a hparam (was hardcoded 3.0)
                coverage_temp = getattr(self.hparams, 'coverage_temperature', 3.0)
                # Compute softer probabilities with a separate temperature (softer than guidance)
                attn_probs_soft = F.softmax(attn_logits / coverage_temp, dim=-1)  # [B, H, N]
                
                # Probabilistic Union (smooth union without gradient gaps)
                union_attn = 1.0 - torch.prod(1.0 - attn_probs_soft, dim=1) # [B, N]
                
                # Normalize the union so it becomes a proper distribution (sum = 1.0)
                union_dist = union_attn / (union_attn.sum(dim=-1, keepdim=True) + 1e-6)
                
                # Normalize the fish mask into an ideal uniform distribution (sum = 1.0)
                mask_area = target_for_loss.sum(dim=-1, keepdim=True).clamp(min=1.0)
                target_dist = target_for_loss / mask_area # [B, N]
                
                # L1 loss between the two distributions
                loss_coverage = F.l1_loss(union_dist, target_dist, reduction='none').sum(dim=-1).mean()

        loss_main, loss_arc, loss_metric = self.main_loss_fn(emb, out["logits"], y)

        self.log("train/loss_arc", loss_arc, batch_size=current_batch_size, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_metric", loss_metric, batch_size=current_batch_size, on_step=False, on_epoch=True, sync_dist=True)

        # --- 3. Segmentation Loss (NEW BLOCK) ---
        loss_segmentation = torch.tensor(0.0, device=self.device)
        if "seg_logits" in out and out["seg_logits"] is not None:
            # Upscale the predicted 14x14 mask to the original image size (e.g., 224x224)
            pred_mask_logits = F.interpolate(
                out["seg_logits"], 
                size=(x.shape[2], x.shape[3]), 
                mode="bilinear", 
                align_corners=False
            )
            
            # Prepare the Ground Truth mask (add channel axis if it's 2D)
            gt_mask = object_mask.float()
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.unsqueeze(1) # [B, 1, H, W]
                
            # Compute standard BCE loss with logits (Sigmoid lives under the hood)
            loss_segmentation = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask)

            
        # --- 4. Total Loss Calculation ---
        attn_lambda = self._get_attention_lambda()
        
        # Pull weights from the config
        diversity_lambda = getattr(self.hparams, "diversity_loss_lambda", 0.2) if attn_lambda > 0 else 0.0
        coverage_lambda = getattr(self.hparams, "coverage_loss_lambda", 0.1) if attn_lambda > 0 else 0.0
        seg_lambda = getattr(self.hparams, "segmentation_loss_lambda", 1.0)

        total_loss = (
            loss_main
            + (attn_lambda * loss_attention_guidance)
            + (diversity_lambda * loss_diversity)
            + (coverage_lambda * loss_coverage)
            + (seg_lambda * loss_segmentation)
        )

        # --- 5. Logging --- (FIX 6: added sync_dist=True for DDP correctness)
        self.log("train/loss", total_loss, batch_size=current_batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_main", loss_main, batch_size=current_batch_size, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_attn_guidance", loss_attention_guidance, batch_size=current_batch_size, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_diversity", loss_diversity, batch_size=current_batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/loss_coverage", loss_coverage, batch_size=current_batch_size, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        
        # Log segmentation artifacts if the module is active
        if loss_segmentation > 0:
            self.log("train/loss_seg", loss_segmentation, batch_size=current_batch_size, on_step=False, on_epoch=True, sync_dist=True)

        self.log("train/lr", self._get_current_lr(), on_step=True, on_epoch=False, prog_bar=True)

        if batch_idx % 50 == 0:
            grad_norm = self._compute_grad_norm()
            if grad_norm is not None:
                self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        # --- 6. Accuracy ---
        with torch.no_grad():
            raw_logits = self.model.arcface_head(emb)
            preds = torch.argmax(raw_logits, dim=1)
            self.train_accuracy.update(preds, y)
            self.train_accuracy_top5.update(raw_logits, y)
            self.train_accuracy_top10.update(raw_logits, y)

        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy_top5", self.train_accuracy_top5, on_step=False, on_epoch=True)
        self.log("train/accuracy_top10", self.train_accuracy_top10, on_step=False, on_epoch=True)

        # ─── LOG ADE COS SCALE (s) AND DYNAMIC MARGIN (m) ───
        # Wrap in try-except in case backbone variable names differ slightly
        try:
            # Look for the ArcFace head (could be named head or arcface_head)
            head = getattr(self.model, 'head', None) or getattr(self.model, 'arcface_head', None)
            
            if head is not None:
                # 1. Log the AdaCos scale (s)
                if hasattr(head, 's'):
                    current_s = head.s.item() if isinstance(head.s, torch.Tensor) else head.s
                    self.log("train/arcface_s", current_s, batch_size=current_batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                
                # 2. Log the dynamic margin (m) while it's enabled!
                if hasattr(head, 'm'):
                    current_m = head.m.item() if isinstance(head.m, torch.Tensor) else head.m
                    self.log("train/arcface_m", current_m, batch_size=current_batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception as e:
            pass # If something goes wrong, training keeps running
            
        return total_loss

    def _compute_grad_norm(self) -> Optional[float]:
        """Compute total gradient L2 norm across all parameters."""
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return None
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2
        )
        return total_norm.item()