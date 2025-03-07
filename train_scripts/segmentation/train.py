import sys

sys.path.insert(0, "/home/fishial/Fishial/clean_repo_4_03/fish-identification")

import os
from datetime import datetime
import fiftyone as fo
import torch

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.tuner import Tuner

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from module.segmentation_mask.model_light import FishSeg
from module.segmentation_mask.dataset import FishialSegmentDatasetFoOnlineCutting


class FineTuneLearningRateFinder(LearningRateFinder):
    """Custom Learning Rate Finder that runs every 'find_range' epochs."""
    def __init__(self, find_range=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.find_range = find_range

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.find_range == 0 or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

#python train_scripts/segmentation/train.py --debug 0.01 --dataset_name test_ds_for_segment_refactoring --num_workers 1 --save_dir /home/fishial/Fishial/TEST_PIPLINE/SEGMENTATION
def get_args():
    parser = ArgumentParser()

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--debug", type=float, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="", help="Name of the FiftyOne dataset")
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 8), help="Number of worker threads for DataLoader")
    parser.add_argument("--log_every_n_steps", type=int, default=100, help="Logging frequency for training")
    parser.add_argument("--find_range", type=int, default=5, help="Frequency of learning rate finder execution")
    parser.add_argument("--encoder_type", type=str, default="FPN", help="Type of encoder to use")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--save_dir", type=str, default="FishialSEGM", help="Directory to save experiment results")

    return parser.parse_args()


def load_datasets(image_size, dataset_name):
    """Load training and validation datasets from FiftyOne."""
    try:
        fo_dataset = fo.load_dataset(dataset_name)
    except ValueError:
        raise ValueError(f"Dataset '{dataset_name}' not found in FiftyOne.")

    train_view = fo_dataset.match(~fo.ViewField("tags").contains("val"))
    val_view = fo_dataset.match_tags("val")

    train_dataset = FishialSegmentDatasetFoOnlineCutting(train_view, aug=False, image_size=image_size)
    valid_dataset = FishialSegmentDatasetFoOnlineCutting(val_view, aug=False, image_size=image_size)

    return train_dataset, valid_dataset


def get_dataloaders(train_dataset, valid_dataset, batch_size, num_workers):
    """Prepare DataLoaders."""
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def setup_trainer(args, save_dir):
    """Configure Lightning Trainer with callbacks and logging."""
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_img_iou",
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="sample-{epoch:02d}-{valid_img_iou:.5f}",
        save_top_k=3,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir, "logs")

    return L.Trainer(
        limit_train_batches=args.debug,
        limit_val_batches=args.debug,
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        precision=16,
        callbacks=[checkpoint_callback, lr_monitor, FineTuneLearningRateFinder(find_range=args.find_range)],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        val_check_interval=0.5
    )


def main():
    args = get_args()

    # Create a unique save directory for the experiment
    experiment_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = os.path.join(args.save_dir, f"{args.encoder_type}_{args.backbone}_{args.image_size}_{experiment_time}")

    # Load datasets and DataLoaders
    train_dataset, valid_dataset = load_datasets(args.image_size, args.dataset_name)
    train_dataloader, valid_dataloader = get_dataloaders(train_dataset, valid_dataset, args.batch_size, args.num_workers)

    # Initialize model
    model = FishSeg(args.encoder_type, args.backbone, in_channels=3, out_classes=1, load_checkpoint=args.ckpt)

    # Setup Trainer
    trainer = setup_trainer(args, save_dir)
    tuner = Tuner(trainer)

    try:
        # Train the model
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    except Exception as e:
        print(f"Training failed with error: {e}")


if __name__ == "__main__":
    main()