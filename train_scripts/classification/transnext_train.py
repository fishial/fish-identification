#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script that reproducibly trains the ArcFace embedding pipeline using a TransNeXt backbone.

We reuse the existing data module / trainer but override the backbone to timm's TransNeXt variants:
`transnext_tiny`, `transnext_small`, `transnext_base` etc.

This mirrors the official DaiShiResearch/TransNeXt work [1], but keeps the same
metric-learning + attention-guided ArcFace head used elsewhere in the repo.

[1]: https://github.com/DaiShiResearch/TransNeXt
"""

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

CURRENT_FILE = os.path.abspath(__file__)
DELIMITER = "fish-identification"
pos = CURRENT_FILE.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FILE[: pos + len(DELIMITER)])

import lightning.pytorch as pl
import timm
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from module.classification_package.src.datamodule import ImageEmbeddingDataModule
from module.classification_package.src.lightning_trainer_fixed import ImageEmbeddingTrainerConvnext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRANSNEXT_WEIGHTS = {
    "transnext_tiny": "https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth",
    "transnext_small": "https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth",
    "transnext_base": "https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth",
}


def get_args():
    parser = argparse.ArgumentParser(description="Train TransNeXt-backed ArcFace embeddings.")

    parser.add_argument("--dataset_name", type=str, required=True, help="FiftyOne dataset name.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--transnext_variant", type=str, choices=["transnext_tiny", "transnext_small", "transnext_base"], default="transnext_small")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--arcface_s", type=float, default=64.0)
    parser.add_argument("--arcface_m", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--classes_per_batch", type=int, default=16)
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr_eta_min", type=float, default=1e-7)
    parser.add_argument("--attention_loss_lambda", type=float, default=0.15)

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--weights_path", type=str, default=None, help="Local path or URL of TransNeXt pretrained weights.")
    parser.add_argument("--download_default_weights", action="store_true", help="Download official TransNeXt weights for the chosen variant.")

    return parser.parse_args()


def _resolve_weights_path(args, output_dir):
    candidate = args.weights_path
    if args.download_default_weights and not candidate:
        candidate = DEFAULT_TRANSNEXT_WEIGHTS.get(args.transnext_variant)
        if not candidate:
            raise ValueError(f"No default weights configured for {args.transnext_variant}")

    if not candidate:
        return None

    if candidate.startswith("http"):
        dest = os.path.join(output_dir, os.path.basename(candidate))
        if not os.path.exists(dest):
            logger.info("Downloading TransNeXt weights from %s", candidate)
            urllib.request.urlretrieve(candidate, dest)
        return dest

    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Specified weights_path does not exist: {candidate}")
    return candidate


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.transnext_variant}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    logger.info("Dataset setup")
    datamodule = ImageEmbeddingDataModule(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    weights_path = _resolve_weights_path(args, run_dir)
    logger.info("Creating TransNeXt backbone (%s)", args.transnext_variant)
    backbone = timm.create_model(args.transnext_variant, pretrained=False, num_classes=0)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            # Some checkpoints nest the actual module
            state = state["model"]
        backbone.load_state_dict(state, strict=False)
    backbone_out_features = getattr(backbone, "num_features", None) or getattr(backbone, "embed_dim", None)
    if backbone_out_features is None:
        raise ValueError("Unable to infer backbone_out_features from TransNeXt model.")

    model = ImageEmbeddingTrainerConvnext(
        num_classes=datamodule.num_classes,
        embedding_dim=args.embedding_dim,
        backbone_model_name=args.transnext_variant,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_eta_min=args.lr_eta_min,
        attention_loss_lambda=args.attention_loss_lambda,
        arcface_s=args.arcface_s,
        arcface_m=args.arcface_m,
        output_dir=run_dir,
        visualize_attention_map=True,
        custom_backbone=backbone,
        backbone_out_features=backbone_out_features,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/accuracy_epoch",
        mode="max",
        save_top_k=3,
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="transnext-{epoch:02d}-{val/accuracy_epoch:.4f}",
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val/accuracy_epoch", patience=20, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger.info("Starting training")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        max_epochs=args.max_epochs,
        logger=TensorBoardLogger(save_dir=run_dir, name="logs"),
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.gradient_accumulation,
        deterministic=False,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
