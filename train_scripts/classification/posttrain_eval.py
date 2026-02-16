#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-training evaluation for Fishial classification models.

What it does:
- Loads a Lightning checkpoint (.ckpt) into the corresponding LightningModule
  (ImageEmbeddingTrainerConvnext / ImageEmbeddingTrainerViT).
- Runs inference on the validation split from ImageEmbeddingDataModule (FiftyOne tags: "val").
- Computes metrics: top-1 / top-5 / top-10 accuracy, macro(top-1) accuracy.
- Saves:
  - metrics.json
  - per_class.csv (support + per-class top-k accuracies)
  - predictions_topk.npz (y_true + topk indices + topk probs)
"""

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch

# Ensure project root (containing "fish-identification") is on sys.path
CURRENT_FILE = os.path.abspath(__file__)
DELIMITER = "fish-identification"
pos = CURRENT_FILE.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FILE[: pos + len(DELIMITER)])

from module.classification_package.src.datamodule import ImageEmbeddingDataModule
from module.classification_package.src.lightning_trainer_fixed import (
    ImageEmbeddingTrainerConvnext,
    ImageEmbeddingTrainerViT,
)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v!r}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_per_class_csv(
    path: str,
    per_class: Dict[int, Dict[str, float]],
    id_to_label: Dict[int, str],
) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_label", "support", "acc_top1", "acc_top5", "acc_top10"])
        for class_id in sorted(per_class.keys()):
            row = per_class[class_id]
            w.writerow(
                [
                    class_id,
                    id_to_label.get(class_id, ""),
                    int(row["support"]),
                    float(row["acc_top1"]),
                    float(row["acc_top5"]),
                    float(row["acc_top10"]),
                ]
            )


@torch.no_grad()
def _run_inference(
    model,
    dataloader,
    device: torch.device,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: [N]
      topk_idx: [N, max_k]
      topk_prob: [N, max_k]
    """
    model.eval()
    model.to(device)

    max_k = max(topk)
    y_true_all = []
    topk_idx_all = []
    topk_prob_all = []

    for batch in dataloader:
        x, y, object_mask = batch
        x = x.to(device, non_blocking=True)
        object_mask = object_mask.to(device, non_blocking=True)

        # LightningModule.forward returns (emb, probs, attn_map) in eval mode
        _, probs, _ = model(x, object_mask=object_mask)
        probs = probs.detach()

        tk_prob, tk_idx = torch.topk(probs, k=max_k, dim=1, largest=True, sorted=True)

        y_true_all.append(y.detach().cpu())
        topk_idx_all.append(tk_idx.cpu())
        topk_prob_all.append(tk_prob.cpu())

    y_true = torch.cat(y_true_all, dim=0).numpy()
    topk_idx = torch.cat(topk_idx_all, dim=0).numpy()
    topk_prob = torch.cat(topk_prob_all, dim=0).numpy()
    return y_true, topk_idx, topk_prob


def _compute_metrics_and_per_class(
    y_true: np.ndarray,
    topk_idx: np.ndarray,
    num_classes: int,
) -> Tuple[dict, Dict[int, Dict[str, float]]]:
    max_k = topk_idx.shape[1]
    k1 = min(1, max_k)
    k5 = min(5, max_k)
    k10 = min(10, max_k)

    correct_top1 = (topk_idx[:, :k1].reshape(-1) == y_true)
    correct_top5 = np.any(topk_idx[:, :k5] == y_true[:, None], axis=1)
    correct_top10 = np.any(topk_idx[:, :k10] == y_true[:, None], axis=1)

    acc1 = float(correct_top1.mean()) if len(y_true) else 0.0
    acc5 = float(correct_top5.mean()) if len(y_true) else 0.0
    acc10 = float(correct_top10.mean()) if len(y_true) else 0.0

    per_class: Dict[int, Dict[str, float]] = {}
    for c in range(num_classes):
        mask = (y_true == c)
        support = int(mask.sum())
        if support == 0:
            continue
        per_class[c] = {
            "support": support,
            "acc_top1": float(correct_top1[mask].mean()),
            "acc_top5": float(correct_top5[mask].mean()),
            "acc_top10": float(correct_top10[mask].mean()),
        }

    # Macro top-1 over classes present in eval split
    macro_acc1 = float(np.mean([v["acc_top1"] for v in per_class.values()])) if per_class else 0.0

    metrics = {
        "n_samples": int(len(y_true)),
        "num_classes": int(num_classes),
        "accuracy_top1": acc1,
        "accuracy_top5": acc5,
        "accuracy_top10": acc10,
        "accuracy_macro_top1": macro_acc1,
    }
    return metrics, per_class


def get_args():
    p = argparse.ArgumentParser(description="Post-training evaluation for fish classification models.")
    p.add_argument("--dataset_name", type=str, required=True, help="FiftyOne dataset name.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .ckpt file.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write eval outputs.")

    p.add_argument("--image_size", type=int, default=None, help="Input image size for eval. If omitted, inferred from backbone name when possible (e.g. *_224/*_384/*_512).")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation dataloader.")
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--backbone_model_name", type=str, default="convnext_small")
    p.add_argument("--embedding_dim", type=int, default=512)
    p.add_argument("--arcface_s", type=float, default=64.0)
    p.add_argument("--arcface_m", type=float, default=0.2)
    p.add_argument("--precision16", type=_str2bool, default=True, help="Use autocast fp16 on CUDA (true/false).")
    return p.parse_args()


def _infer_image_size_from_backbone_name(backbone_model_name: str) -> int | None:
    # timm naming convention often ends with *_224, *_384, *_512 for fixed-resolution variants
    for s in (512, 384, 256, 224):
        if backbone_model_name.endswith(f"_{s}"):
            return s
    return None


def _validate_maxvit_resolution(backbone_model_name: str, image_size: int) -> None:
    """
    MaxViT/MaxxViT models can have strict window partition constraints.
    The most reliable choice is using the resolution the model variant was trained/configured for (suffix *_224/*_384/*_512).
    """
    name = backbone_model_name.lower()
    if ("maxvit" in name or "maxxvit" in name) and image_size is not None:
        inferred = _infer_image_size_from_backbone_name(backbone_model_name)
        if inferred is not None and inferred != image_size:
            raise ValueError(
                f"Incompatible --image_size {image_size} for backbone '{backbone_model_name}'. "
                f"This variant is typically fixed to {inferred}. "
                f"Use --image_size {inferred}, or switch backbone to a matching variant (e.g. *_384) if you want 384."
            )


def main():
    args = get_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    _ensure_dir(out_dir)

    # Setup datamodule (val split)
    if args.image_size is None:
        args.image_size = _infer_image_size_from_backbone_name(args.backbone_model_name) or 224
    _validate_maxvit_resolution(args.backbone_model_name, args.image_size)

    dm = ImageEmbeddingDataModule(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        classes_per_batch=1,   # unused for val loader
        samples_per_class=1,   # unused for val loader
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    dm.setup()
    val_loader = dm.val_dataloader()

    # Build model module (same logic as training script)
    model_cls = ImageEmbeddingTrainerConvnext if "convnext" in args.backbone_model_name else ImageEmbeddingTrainerViT
    model = model_cls(
        num_classes=dm.num_classes,
        embedding_dim=args.embedding_dim,
        backbone_model_name=args.backbone_model_name,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        lr=1e-4,
        weight_decay=0.0,
        lr_eta_min=1e-7,
        attention_loss_lambda=0.0,
        load_checkpoint=args.checkpoint,
        output_dir=out_dir,
        visualize_attention_map=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.precision16) and device.type == "cuda"

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        y_true, topk_idx, topk_prob = _run_inference(model, val_loader, device=device, topk=(1, 5, 10))

    metrics, per_class = _compute_metrics_and_per_class(y_true, topk_idx, num_classes=dm.num_classes)

    # id -> label mapping (invert dm.label_to_id)
    id_to_label = {idx: label for label, idx in dm.label_to_id.items()}

    # Save outputs
    _save_json(
        os.path.join(out_dir, "metrics.json"),
        {
            "checkpoint": os.path.abspath(args.checkpoint),
            "dataset_name": args.dataset_name,
            "backbone_model_name": args.backbone_model_name,
            "embedding_dim": args.embedding_dim,
            "arcface_s": args.arcface_s,
            "arcface_m": args.arcface_m,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "amp_fp16": bool(use_amp),
            "metrics": metrics,
        },
    )
    _write_per_class_csv(os.path.join(out_dir, "per_class.csv"), per_class, id_to_label=id_to_label)
    np.savez_compressed(
        os.path.join(out_dir, "predictions_topk.npz"),
        y_true=y_true.astype(np.int64),
        topk_idx=topk_idx.astype(np.int64),
        topk_prob=topk_prob.astype(np.float32),
    )

    print(f"[eval] Wrote results to: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

