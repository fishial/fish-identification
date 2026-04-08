# -*- coding: utf-8 -*-
"""
Automatic hyperparameter tuning based on training metrics analysis.

Three modes of operation:

1. **Between-runs** — ``AutoTuner`` reads metrics + config from the previous
   run, diagnoses overfitting / over-regularisation / plateau, and proposes
   adjusted hyperparameters.  Accepts *cross-run context* (how many
   consecutive runs had no improvement) so it can apply escalating
   plateau-breaking strategies.

2. **Intra-training** — ``AdaptiveRegularizationCallback`` monitors the
   train/val gap during a single run and dynamically adjusts dropout +
   weight decay.

3. **Post-training** — ``generate_next_run_suggestion`` writes a ready-to-run
   shell command into ``next_run_suggestion.sh``.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

AUGMENTATION_LEVELS = ["basic", "standard", "medium", "strong"]


# ============================================================================
# Diagnosis
# ============================================================================

@dataclass
class TrainingDiagnosis:
    overfitting_severity: float = 0.0   # 0→1
    over_regularized: bool = False      # train ≈ val but both plateau
    underfitting_severity: float = 0.0
    stagnation_epochs: int = 0
    best_val_accuracy: float = 0.0
    best_val_epoch: int = 0
    total_epochs: int = 0
    final_train_accuracy: float = 0.0
    final_val_accuracy: float = 0.0
    overfit_gap: float = 0.0
    val_trend: str = "unknown"
    loss_trend: str = "unknown"
    messages: List[str] = field(default_factory=list)


# ============================================================================
# AutoTuner
# ============================================================================

class AutoTuner:

    def __init__(self, previous_run_dir: str):
        self.previous_run_dir = previous_run_dir
        self.metrics_path = os.path.join(previous_run_dir, "metrics_history.jsonl")
        self.config_path = os.path.join(previous_run_dir, "config.json")

    # ------------------------------------------------------------------ io

    def load_metrics(self) -> List[Dict]:
        if not os.path.exists(self.metrics_path):
            raise FileNotFoundError(f"Metrics not found: {self.metrics_path}")
        out: List[Dict] = []
        with open(self.metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def load_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return json.load(f)

    # -------------------------------------------------------------- diagnose

    @staticmethod
    def diagnose(metrics: List[Dict]) -> TrainingDiagnosis:
        diag = TrainingDiagnosis()
        if not metrics:
            diag.messages.append("No metrics found")
            return diag

        val_accs = [float(m["val/accuracy_epoch"]) for m in metrics if "val/accuracy_epoch" in m]
        train_accs = [float(m["train/accuracy_epoch"]) for m in metrics if "train/accuracy_epoch" in m]
        val_losses = [float(m["val/loss"]) for m in metrics if "val/loss" in m]

        diag.total_epochs = len(metrics)
        if not val_accs:
            diag.messages.append("No validation accuracy found")
            return diag

        diag.best_val_accuracy = max(val_accs)
        diag.best_val_epoch = val_accs.index(diag.best_val_accuracy)
        diag.final_val_accuracy = val_accs[-1]
        diag.final_train_accuracy = train_accs[-1] if train_accs else 0.0

        # --- gap ---
        w = min(5, len(val_accs))
        avg_val = sum(val_accs[-w:]) / w
        avg_train = sum(train_accs[-w:]) / w if train_accs else 0.0
        diag.overfit_gap = avg_train - avg_val

        if diag.overfit_gap <= 0.015:
            diag.overfitting_severity = 0.0
        elif diag.overfit_gap <= 0.03:
            diag.overfitting_severity = 0.25
            diag.messages.append(f"Mild overfitting: gap={diag.overfit_gap:.4f}")
        elif diag.overfit_gap <= 0.05:
            diag.overfitting_severity = 0.5
            diag.messages.append(f"Moderate overfitting: gap={diag.overfit_gap:.4f}")
        elif diag.overfit_gap <= 0.08:
            diag.overfitting_severity = 0.75
            diag.messages.append(f"Significant overfitting: gap={diag.overfit_gap:.4f}")
        else:
            diag.overfitting_severity = 1.0
            diag.messages.append(f"Severe overfitting: gap={diag.overfit_gap:.4f}")

        # --- over-regularised ---
        # train_acc ≈ val_acc AND train_acc < 0.98 → model can't learn enough
        if train_accs and diag.overfit_gap < 0.01 and avg_train < 0.98:
            diag.over_regularized = True
            diag.messages.append(
                f"Over-regularised: train={avg_train:.4f}, val={avg_val:.4f}, "
                f"gap={diag.overfit_gap:.4f} — model capacity suppressed"
            )

        # --- stagnation ---
        diag.stagnation_epochs = len(val_accs) - 1 - diag.best_val_epoch
        if diag.stagnation_epochs > 10:
            diag.messages.append(
                f"Stagnation: no improvement for {diag.stagnation_epochs} epochs "
                f"(best={diag.best_val_accuracy:.4f} @ epoch {diag.best_val_epoch})"
            )

        # --- val trend ---
        tw = min(10, len(val_accs))
        if tw >= 4:
            recent = val_accs[-tw:]
            half = tw // 2
            h1 = sum(recent[:half]) / half
            h2 = sum(recent[half:]) / (tw - half)
            if h2 > h1 + 0.002:
                diag.val_trend = "improving"
            elif h2 < h1 - 0.002:
                diag.val_trend = "declining"
                diag.messages.append("Validation accuracy declining")
            else:
                diag.val_trend = "plateau"
                diag.messages.append("Validation accuracy plateaued")

        # --- loss trend ---
        if len(val_losses) >= 4:
            lw = min(10, len(val_losses))
            rl = val_losses[-lw:]
            lh = lw // 2
            if sum(rl[lh:]) / (lw - lh) > sum(rl[:lh]) / lh + 0.005:
                diag.loss_trend = "worsening"
            elif sum(rl[lh:]) / (lw - lh) < sum(rl[:lh]) / lh - 0.005:
                diag.loss_trend = "improving"
            else:
                diag.loss_trend = "stable"

        # --- underfitting ---
        if diag.final_train_accuracy < 0.90 and diag.final_val_accuracy < 0.85:
            diag.underfitting_severity = 1.0 - diag.final_train_accuracy
            diag.messages.append(
                f"Underfitting: train={diag.final_train_accuracy:.4f}, "
                f"val={diag.final_val_accuracy:.4f}"
            )

        return diag

    # ------------------------------------------------- single-run adjustments

    @staticmethod
    def suggest_adjustments(
        diag: TrainingDiagnosis,
        prev_config: Dict,
    ) -> Tuple[Dict[str, Any], List[str]]:
        adj: Dict[str, Any] = {}
        reasons: List[str] = []
        sev = diag.overfitting_severity

        # ── OVERFITTING ──
        if sev > 0:
            cur = prev_config.get("embedding_dropout_rate") or 0.0
            delta = {0.25: 0.03, 0.5: 0.05, 0.75: 0.07, 1.0: 0.09}.get(sev, 0.0)
            if delta:
                new = round(min(cur + delta, 0.4), 3)
                if new != cur:
                    adj["embedding_dropout_rate"] = new
                    reasons.append(f"dropout: {cur} → {new}")

            cur_wd = prev_config.get("weight_decay", 0.05)
            mult = {0.25: 1.15, 0.5: 1.3, 0.75: 1.5, 1.0: 1.7}.get(sev, 1.0)
            new_wd = round(min(cur_wd * mult, 0.2), 5)
            if new_wd != cur_wd:
                adj["weight_decay"] = new_wd
                reasons.append(f"weight_decay: {cur_wd} → {new_wd}")

            cur_ls = prev_config.get("label_smoothing", 0.1)
            ls_d = {0.25: 0.02, 0.5: 0.04, 0.75: 0.06, 1.0: 0.08}.get(sev, 0.0)
            new_ls = round(min(cur_ls + ls_d, 0.25), 3)
            if new_ls != cur_ls:
                adj["label_smoothing"] = new_ls
                reasons.append(f"label_smoothing: {cur_ls} → {new_ls}")

            cur_aug = prev_config.get("augmentation_preset", "standard")
            if cur_aug in AUGMENTATION_LEVELS and sev >= 0.5:
                idx = AUGMENTATION_LEVELS.index(cur_aug)
                if idx < len(AUGMENTATION_LEVELS) - 1:
                    adj["augmentation_preset"] = AUGMENTATION_LEVELS[idx + 1]
                    reasons.append(f"augmentation: {cur_aug} → {adj['augmentation_preset']}")

            cur_lr = prev_config.get("learning_rate", 1e-4)
            lr_m = {0.25: 0.85, 0.5: 0.7, 0.75: 0.5, 1.0: 0.35}.get(sev, 1.0)
            new_lr = cur_lr * lr_m
            if new_lr != cur_lr:
                adj["learning_rate"] = new_lr
                reasons.append(f"learning_rate: {cur_lr:.2e} → {new_lr:.2e}")

            cur_m = prev_config.get("arcface_m", 0.2)
            if sev >= 0.75 and cur_m > 0.15:
                new_m = round(max(cur_m - 0.05, 0.1), 2)
                adj["arcface_m"] = new_m
                reasons.append(f"arcface_m: {cur_m} → {new_m}")

            if prev_config.get("use_swa") and sev >= 0.5:
                cs = prev_config.get("swa_epoch_start", 0.75)
                ns = round(max(cs - 0.15, 0.3), 2)
                if ns != cs:
                    adj["swa_epoch_start"] = ns
                    reasons.append(f"swa_epoch_start: {cs} → {ns}")

        # ── OVER-REGULARISED ──
        if diag.over_regularized:
            cur_drop = prev_config.get("embedding_dropout_rate") or 0.0
            if cur_drop > 0.08:
                new_d = round(max(cur_drop - 0.04, 0.05), 3)
                adj["embedding_dropout_rate"] = new_d
                reasons.append(f"dropout (over-reg): {cur_drop} → {new_d}")

            cur_wd = prev_config.get("weight_decay", 0.05)
            if cur_wd > 0.03:
                new_wd = round(cur_wd * 0.7, 5)
                adj["weight_decay"] = new_wd
                reasons.append(f"weight_decay (over-reg): {cur_wd} → {new_wd}")

            cur_ls = prev_config.get("label_smoothing", 0.1)
            if cur_ls > 0.03:
                new_ls = round(max(cur_ls - 0.03, 0.02), 3)
                adj["label_smoothing"] = new_ls
                reasons.append(f"label_smoothing (over-reg): {cur_ls} → {new_ls}")

            cur_aug = prev_config.get("augmentation_preset", "standard")
            if cur_aug in AUGMENTATION_LEVELS:
                idx = AUGMENTATION_LEVELS.index(cur_aug)
                if idx > 1:
                    adj["augmentation_preset"] = AUGMENTATION_LEVELS[idx - 1]
                    reasons.append(f"augmentation (over-reg): {cur_aug} → {adj['augmentation_preset']}")

            cur_lr = prev_config.get("learning_rate", 1e-4)
            adj["learning_rate"] = cur_lr * 1.4
            reasons.append(f"learning_rate (over-reg): {cur_lr:.2e} → {cur_lr*1.4:.2e}")

        # ── UNDERFITTING ──
        if diag.underfitting_severity > 0.3:
            cur_lr = prev_config.get("learning_rate", 1e-4)
            adj["learning_rate"] = cur_lr * 1.5
            reasons.append(f"learning_rate (underfit): {cur_lr:.2e} → {cur_lr*1.5:.2e}")
            for k, lo in [("embedding_dropout_rate", 0.0), ("weight_decay", 0.01), ("label_smoothing", 0.0)]:
                cur = prev_config.get(k) or 0.0
                if cur > lo + 0.02:
                    nv = round(max(cur * 0.6, lo), 5)
                    adj[k] = nv
                    reasons.append(f"{k} (underfit): {cur} → {nv}")

        # ── STAGNATION (within single run) ──
        if diag.stagnation_epochs > 15 and sev < 0.25:
            if not prev_config.get("use_cyclic_lr"):
                adj["use_cyclic_lr"] = True
                adj["cyclic_mode"] = "warm_restarts"
                adj["cyclic_t0"] = 5
                reasons.append("Enabling cyclic LR for plateau")
            else:
                cur_t0 = prev_config.get("cyclic_t0", 10)
                if cur_t0 > 3:
                    adj["cyclic_t0"] = max(cur_t0 - 2, 3)
                    reasons.append(f"cyclic_t0: {cur_t0} → {adj['cyclic_t0']}")

        return adj, reasons

    # ----------------------------------------- cross-run plateau breaking

    @staticmethod
    def suggest_plateau_adjustments(
        prev_config: Dict,
        stale_runs: int,
        best_val_overall: float,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Escalating strategies when val accuracy plateaus across multiple runs.

        Each level tries a qualitatively different approach — not just
        tweaking regularisation, but changing batch geometry, loss dynamics,
        memory, attention, etc.

        For a ~775-class fish dataset the biggest untapped levers are:
        - batch geometry (more classes per batch → better hard-negative mining)
        - metric/arcface loss balance
        - cross-batch memory
        - attention_loss_lambda
        """
        adj: Dict[str, Any] = {}
        reasons: List[str] = []

        def _get(k, default=None):
            return prev_config.get(k, default)

        # ── Level 1: widen batch + ease regularisation ──
        if stale_runs == 1:
            # More classes per batch = more negatives for metric learning
            cur_cpb = _get("classes_per_batch", 32)
            cur_spc = _get("samples_per_class", 6)
            if cur_cpb <= 40 and cur_spc >= 5:
                adj["classes_per_batch"] = 48
                adj["samples_per_class"] = 4
                reasons.append(
                    f"Batch geometry: {cur_cpb}×{cur_spc}→48×4 "
                    f"(more class diversity for mining)"
                )

            # Reduce attention guidance that constrains features
            cur_attn = _get("attention_loss_lambda", 0.15)
            if cur_attn > 0.05:
                adj["attention_loss_lambda"] = round(cur_attn * 0.5, 3)
                reasons.append(
                    f"attention_loss_lambda: {cur_attn} → {adj['attention_loss_lambda']} "
                    f"(less feature constraint)"
                )

            # Ease regularisation
            cur_drop = _get("embedding_dropout_rate") or 0.0
            if cur_drop > 0.08:
                adj["embedding_dropout_rate"] = round(max(cur_drop - 0.04, 0.05), 3)
                reasons.append(f"dropout: {cur_drop} → {adj['embedding_dropout_rate']}")

            cur_wd = _get("weight_decay", 0.05)
            if cur_wd > 0.03:
                adj["weight_decay"] = round(cur_wd * 0.75, 5)
                reasons.append(f"weight_decay: {cur_wd} → {adj['weight_decay']}")

            cur_lr = _get("learning_rate", 1e-4)
            adj["learning_rate"] = cur_lr * 1.4
            reasons.append(f"learning_rate: {cur_lr:.2e} → {cur_lr*1.4:.2e}")

            cur_aug = _get("augmentation_preset", "standard")
            if cur_aug == "strong":
                adj["augmentation_preset"] = "medium"
                reasons.append("augmentation: strong → medium")

        # ── Level 2: loss dynamics + even wider batch ──
        elif stale_runs == 2:
            # Even wider class coverage in batch
            cur_cpb = _get("classes_per_batch", 32)
            cur_spc = _get("samples_per_class", 6)
            adj["classes_per_batch"] = 64
            adj["samples_per_class"] = 4
            adj["accumulate_grad_batches"] = 2
            reasons.append(
                f"Batch geometry: {cur_cpb}×{cur_spc}→64×4 "
                f"(8% of classes per batch, accum=2)"
            )

            # Shift weight toward metric learning
            cur_mw = _get("metric_weight", 0.1)
            cur_aw = _get("arcface_weight", 0.9)
            adj["metric_weight"] = round(min(cur_mw + 0.1, 0.4), 2)
            adj["arcface_weight"] = round(max(cur_aw - 0.1, 0.6), 2)
            reasons.append(
                f"Loss balance: metric {cur_mw}→{adj['metric_weight']}, "
                f"arcface {cur_aw}→{adj['arcface_weight']}"
            )

            # Enable cross-batch memory — keeps embeddings from recent batches
            if not _get("use_cross_batch_memory", False):
                adj["use_cross_batch_memory"] = True
                adj["memory_size"] = 4096
                reasons.append("Enabling cross-batch memory (4096)")

            # Reduce attention constraint further
            cur_attn = _get("attention_loss_lambda", 0.15)
            adj["attention_loss_lambda"] = round(min(cur_attn, 0.05), 3)
            reasons.append(f"attention_loss_lambda: {cur_attn} → {adj['attention_loss_lambda']}")

            cur_lr = _get("learning_rate", 1e-4)
            adj["learning_rate"] = cur_lr * 1.5
            reasons.append(f"learning_rate: {cur_lr:.2e} → {cur_lr*1.5:.2e}")

            cur_t0 = _get("cyclic_t0", 10)
            adj["cyclic_t0"] = max(cur_t0 - 2, 3)
            reasons.append(f"cyclic_t0: {cur_t0} → {adj['cyclic_t0']}")

            cur_fw = _get("focal_weight", 0.0)
            if cur_fw < 0.15:
                adj["focal_weight"] = round(cur_fw + 0.05, 3)
                reasons.append(f"focal_weight: {cur_fw} → {adj['focal_weight']}")

        # ── Level 3: full shake-up ──
        elif stale_runs == 3:
            adj["classes_per_batch"] = 64
            adj["samples_per_class"] = 4
            adj["accumulate_grad_batches"] = 2
            reasons.append("Batch: 64×4, accum=2")

            adj["metric_weight"] = 0.3
            adj["arcface_weight"] = 0.7
            reasons.append("Loss: metric=0.3, arcface=0.7")

            adj["use_cross_batch_memory"] = True
            adj["memory_size"] = 8192
            reasons.append("Cross-batch memory: 8192")

            adj["attention_loss_lambda"] = 0.02
            reasons.append("attention_loss_lambda → 0.02 (minimal)")

            adj["embedding_dropout_rate"] = 0.1
            adj["weight_decay"] = 0.04
            adj["label_smoothing"] = 0.04
            adj["augmentation_preset"] = "medium"
            reasons.append("Regularisation: moderate reset")

            cur_lr = _get("learning_rate", 1e-4)
            adj["learning_rate"] = cur_lr * 2.0
            reasons.append(f"learning_rate: {cur_lr:.2e} → {cur_lr*2:.2e}")

            adj["cyclic_t0"] = 4
            reasons.append("cyclic_t0 → 4")

            cur_K = _get("arcface_K", 3)
            if cur_K < 5:
                adj["arcface_K"] = min(cur_K + 1, 5)
                reasons.append(f"arcface_K: {cur_K} → {adj['arcface_K']}")

        # ── Level 4+: extreme diversity ──
        elif stale_runs >= 4:
            # Try maximum batch diversity with smaller effective batch
            adj["classes_per_batch"] = 96
            adj["samples_per_class"] = 3
            adj["accumulate_grad_batches"] = 2
            reasons.append("Batch: 96×3 (12% of classes per batch!), accum=2")

            adj["metric_weight"] = 0.35
            adj["arcface_weight"] = 0.65
            adj["focal_weight"] = 0.15
            reasons.append("Loss: metric=0.35, arcface=0.65, focal=0.15")

            adj["use_cross_batch_memory"] = True
            adj["memory_size"] = 8192
            reasons.append("Cross-batch memory: 8192")

            adj["attention_loss_lambda"] = 0.01
            reasons.append("attention_loss_lambda → 0.01")

            adj["embedding_dropout_rate"] = 0.08
            adj["weight_decay"] = 0.035
            adj["label_smoothing"] = 0.03
            adj["augmentation_preset"] = "medium"
            reasons.append("Regularisation: light")

            cur_lr = _get("learning_rate", 1e-4)
            adj["learning_rate"] = max(cur_lr * 2.5, 3e-5)
            reasons.append(f"learning_rate → {adj['learning_rate']:.2e}")

            adj["cyclic_t0"] = 3
            adj["arcface_K"] = 5
            adj["arcface_m"] = 0.25
            adj["arcface_m_start"] = 0.08
            reasons.append("ArcFace: K=5, m=0.25, m_start=0.08, t0=3")

        return adj, reasons

    # -------------------------------------------------------------- main

    def run(
        self,
        stale_runs: int = 0,
        best_val_overall: float = 0.0,
    ) -> Tuple[Dict[str, Any], TrainingDiagnosis, List[str]]:
        metrics = self.load_metrics()
        prev_config = self.load_config()
        diag = self.diagnose(metrics)

        # Single-run diagnosis-based adjustments
        adjustments, reasons = self.suggest_adjustments(diag, prev_config)

        # Cross-run plateau-breaking: when multiple consecutive runs fail
        # to improve, apply escalating strategies ON TOP of single-run fixes.
        # Build a merged config so the plateau strategy sees the full picture.
        if stale_runs >= 1:
            merged_config = dict(prev_config)
            merged_config.update(adjustments)
            p_adj, p_reasons = self.suggest_plateau_adjustments(
                merged_config, stale_runs, best_val_overall,
            )
            adjustments.update(p_adj)
            reasons.extend(p_reasons)

        return adjustments, diag, reasons

    def apply_to_args(self, args) -> TrainingDiagnosis:
        adjustments, diag, reasons = self.run()
        if not reasons:
            logger.info("[AutoTuner] No adjustments needed.")
            return diag
        logger.info("=" * 70)
        logger.info("[AutoTuner] Diagnosis (%s):", self.previous_run_dir)
        for msg in diag.messages:
            logger.info("  • %s", msg)
        logger.info("  best_val=%.4f (epoch %d), gap=%.4f",
                     diag.best_val_accuracy, diag.best_val_epoch, diag.overfit_gap)
        logger.info("[AutoTuner] %d adjustment(s):", len(reasons))
        for r in reasons:
            logger.info("  → %s", r)
        logger.info("=" * 70)
        for k, v in adjustments.items():
            setattr(args, k, v)
        return diag


# ============================================================================
# Adaptive regularisation callback
# ============================================================================

try:
    import lightning.pytorch as pl
    _LightningCallback = pl.Callback
except ImportError:
    import pytorch_lightning as pl  # type: ignore[no-redef]
    _LightningCallback = pl.Callback


class AdaptiveRegularizationCallback(_LightningCallback):
    """
    Monitors train/val gap during training and dynamically increases
    dropout + weight decay when overfitting is detected.
    """

    def __init__(
        self,
        gap_threshold: float = 0.025,
        patience: int = 3,
        max_dropout: float = 0.40,
        max_wd_multiplier: float = 2.5,
        check_every_n_epochs: int = 3,
    ):
        super().__init__()
        self.gap_threshold = gap_threshold
        self.patience = patience
        self.max_dropout = max_dropout
        self.max_wd_multiplier = max_wd_multiplier
        self.check_every_n_epochs = check_every_n_epochs
        self._overfit_counter = 0
        self._initial_wd: Optional[float] = None

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        ta, va = m.get("train/accuracy_epoch"), m.get("val/accuracy_epoch")
        if ta is None or va is None:
            return
        gap = float(ta) - float(va)
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return
        if self._initial_wd is None and trainer.optimizers:
            self._initial_wd = trainer.optimizers[0].param_groups[0].get("weight_decay", 0.05)
        if gap > self.gap_threshold:
            self._overfit_counter += 1
        else:
            self._overfit_counter = max(0, self._overfit_counter - 1)
        if self._overfit_counter >= self.patience:
            self._adjust(trainer, pl_module, gap)
            self._overfit_counter = 0

    def _adjust(self, trainer, pl_module, gap):
        model = pl_module.model
        fc = getattr(model, "embedding_fc", None)
        if fc:
            for layer in fc:
                if isinstance(layer, torch.nn.Dropout):
                    old_p = layer.p
                    layer.p = min(old_p + 0.02, self.max_dropout)
                    if layer.p != old_p:
                        logger.info("[AdaptiveReg] epoch %d: dropout %.3f→%.3f (gap=%.4f)",
                                    trainer.current_epoch, old_p, layer.p, gap)
                    break
        if self._initial_wd and trainer.optimizers:
            mx = self._initial_wd * self.max_wd_multiplier
            for pg in trainer.optimizers[0].param_groups:
                old = pg["weight_decay"]
                pg["weight_decay"] = min(old * 1.15, mx)


# ============================================================================
# Helpers
# ============================================================================

def find_best_checkpoint(run_dir: str) -> Optional[str]:
    """
    Scans the checkpoints directory and returns the path to the model 
    with the highest accuracy based on the filename.
    
    Expected filename format: model-epoch17-acc9610.ckpt
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    candidates: List[Tuple[float, str]] = []
    
    # Walk through the directory to find all .ckpt files
    for root, _, files in os.walk(ckpt_dir):
        for fn in files:
            # Skip non-checkpoint files and the generic 'last.ckpt'
            if not fn.endswith(".ckpt") or fn == "last.ckpt":
                continue
            
            full_path = os.path.join(root, fn)
            try:
                # Extract the numeric part after 'acc'
                # Example: 'model-epoch17-acc9610.ckpt' -> '9610'
                acc_str = fn.split("acc")[-1].replace(".ckpt", "")
                
                # Convert to float for comparison (e.g., 9610 becomes 0.9610)
                # We divide by 10000.0 to match the scaling used in your trainer
                accuracy = float(acc_str) / 10000.0
            except (ValueError, IndexError):
                # If parsing fails, treat accuracy as zero
                accuracy = 0.0
                
            candidates.append((accuracy, full_path))

    if candidates:
        # Sort candidates by accuracy in descending order
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_model_path = candidates[0][1]
        best_accuracy = candidates[0][0]
        
        print(f"[Best Checkpoint] Found: {os.path.basename(best_model_path)} with accuracy: {best_accuracy:.4f}")
        return best_model_path

    # Fallback to last.ckpt if no versioned checkpoints are found
    last_path = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_path):
        print("[Best Checkpoint] No specific accuracy checkpoints found. Returning last.ckpt")
        return last_path

    return None


def generate_next_run_suggestion(args, output_dir: str) -> None:
    """Write next_run_suggestion.sh with auto-tuned command."""
    try:
        tuner = AutoTuner(output_dir)
        adjustments, diag, reasons = tuner.run()
    except Exception as e:
        logger.warning("[AutoTuner] Suggestion failed: %s", e)
        return
    if not reasons:
        logger.info("[AutoTuner] No adjustments suggested (val=%.4f, gap=%.4f).",
                     diag.best_val_accuracy, diag.overfit_gap)
        return

    next_args = vars(args).copy()
    for k, v in adjustments.items():
        next_args[k] = v
    best_ckpt = find_best_checkpoint(output_dir)
    if best_ckpt:
        next_args["load_weights_from_checkpoint"] = best_ckpt

    skip = {"resume_from_checkpoint", "limit_train_batches", "limit_val_batches",
            "cache_dir", "no_cache", "auto_tune_from", "use_adaptive_regularization"}
    parts = ["python lightning_train.py"]
    for k in sorted(next_args):
        if k in skip or k.startswith("_"):
            continue
        v = next_args[k]
        if v is None:
            continue
        if k == "output_dir":
            v = os.path.dirname(output_dir)
        if isinstance(v, bool):
            parts.append(f"  --{k} {'true' if v else 'false'}")
        elif isinstance(v, list):
            parts.append(f'  --{k} "{",".join(str(x) for x in v)}"')
        elif isinstance(v, float):
            parts.append(f"  --{k} {v:g}")
        else:
            parts.append(f"  --{k} {v}")
    cmd = " \\\n".join(parts)

    logger.info("\n" + "=" * 70)
    logger.info("[AutoTuner] NEXT RUN SUGGESTION")
    for msg in diag.messages:
        logger.info("  • %s", msg)
    for r in reasons:
        logger.info("  → %s", r)
    logger.info("\n%s\n", cmd)
    logger.info("=" * 70)

    path = os.path.join(output_dir, "next_run_suggestion.sh")
    with open(path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Previous run: {output_dir}\n")
        f.write(f"# Best val: {diag.best_val_accuracy:.4f} @ epoch {diag.best_val_epoch}\n#\n")
        for r in reasons:
            f.write(f"#  → {r}\n")
        f.write(f"\n{cmd}\n")
    logger.info("Saved → %s", path)


def resolve_run_dir(path: str) -> str:
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "config.json")):
            return path
        subs = [os.path.join(path, d) for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
                and os.path.exists(os.path.join(path, d, "config.json"))]
        if subs:
            subs.sort(key=os.path.getmtime, reverse=True)
            logger.info("[AutoTuner] Resolved latest: %s", subs[0])
            return subs[0]
    raise FileNotFoundError(f"Cannot resolve run dir from: {path}")
