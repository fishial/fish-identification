import os
from datetime import datetime
import fiftyone as fo
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.tuner import Tuner

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from module.segmentation_mask.model_light import FishSeg
from module.segmentation_mask.dataset import FishialSegmentDatasetFoOnlineCuting


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


def get_args():
    parser = ArgumentParser()

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--debug", type=float, default=None)
    parser.add_argument("--ckpt", type=str, default=None)

    return parser.parse_args()


def load_datasets(image_size):
    """Load training and validation datasets from FiftyOne."""
    fo_dataset = fo.load_dataset("SEGM-2024-V0.8")

    train_view = fo_dataset.match(~fo.ViewField("tags").contains("val"))
    val_view = fo_dataset.match_tags("val")

    train_dataset = FishialSegmentDatasetFoOnlineCuting(train_view, aug=False, image_size=image_size)
    valid_dataset = FishialSegmentDatasetFoOnlineCuting(val_view, aug=False, image_size=image_size)

    return train_dataset, valid_dataset


def get_dataloaders(train_dataset, valid_dataset, batch_size=16):
    """Prepare DataLoaders."""
    num_workers = min(os.cpu_count(), 8)  # Avoid excessive worker threads
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
        callbacks=[checkpoint_callback, lr_monitor, FineTuneLearningRateFinder(find_range=5)],
        logger=logger,
        log_every_n_steps=100,
        deterministic=True,
    )


def main():
    args = get_args()

    # Create a unique save directory for the experiment
    experiment_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    encoder_type, backbone = "FPN", "resnet18"
    save_dir = os.path.join("FishialSEGM", f"{encoder_type}_{backbone}_{args.image_size}_{experiment_time}")

    # Load datasets and DataLoaders
    train_dataset, valid_dataset = load_datasets(args.image_size)
    train_dataloader, valid_dataloader = get_dataloaders(train_dataset, valid_dataset)

    # Initialize model
    model = FishSeg(encoder_type, backbone, in_channels=3, out_classes=1, load_checkpoint=args.ckpt)

    # Setup Trainer
    trainer = setup_trainer(args, save_dir)
    tuner = Tuner(trainer)

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    main()