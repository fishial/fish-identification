import os
import sys
import json
import logging
import argparse
from datetime import datetime

import torch
import torchvision.models as models
import fiftyone as fo
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import samplers
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data.sampler import BatchSampler

# Apex for mixed precision training
from apex import amp

# Import custom modules
sys.path.insert(1, '')  # Change path if needed
from module.classification_package.src.utils import (
    WarmupCosineSchedule, find_device, save_json, get_data_config
)
from module.classification_package.src.model import init_model
from module.classification_package.src.dataset import FishialDatasetFoOnlineCuting
from module.classification_package.src.train import train


# ============== CONFIGURATION ==============
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a classification model.")

    # Dataset and paths
    parser.add_argument("--dataset", type=str, required=True, help="FiftyOne dataset name")
    parser.add_argument("--output_dir", type=str, default="/home/fishial/Fishial/output/classification",
                        help="Directory to save training outputs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=150, help="Batch size for training")
    parser.add_argument("--classes_per_batch", type=int, default=30, help="Number of classes per batch")
    parser.add_argument("--samples_per_class", type=int, default=5, help="Number of samples per class")
    parser.add_argument("--lr", type=float, default=3e-2, help="Initial learning rate")

    # Model and optimizer settings
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Backbone model name")
    parser.add_argument("--embedding_size", type=int, default=128, help="Size of embedding layer")

    return parser.parse_args()


def prepare_dataset(dataset_name):
    """Load FiftyOne dataset and split into train/val."""
    logger.info(f"Loading FiftyOne dataset: {dataset_name}")
    fo_dataset = fo.load_dataset(dataset_name)
    train_data = fo_dataset.match_tags("train")
    val_data = fo_dataset.match_tags("val")

    train_records = get_data_config(train_data)
    val_records = get_data_config(val_data)

    label_to_id = {label: idx for idx, label in enumerate(train_records)}
    id_to_label = {idx: label for idx, label in enumerate(train_records)}

    return train_records, val_records, label_to_id, id_to_label


def prepare_dataloader(train_records, val_records, label_to_id, batch_size, classes_per_batch, samples_per_class):
    """Prepare DataLoader with balanced batch sampling."""
    logger.info("Initializing dataset and DataLoader...")

    train_dataset = FishialDatasetFoOnlineCuting(
        train_records, label_to_id, train_state=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), Image.BILINEAR),
            torchvision.transforms.RandomAutocontrast(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomErasing(p=0.358, scale=(0.05, 0.4), ratio=(0.05, 6.1), value=0),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        crop_type='rect'
    )
    
    val_dataset = FishialDatasetFoOnlineCuting(
        val_records, label_to_id,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        crop_type='rect'
    )

    logger.info(f"Number of training classes: {train_dataset.n_classes}")
    logger.info(f"Number of validation classes: {val_dataset.n_classes}")

    sampler = MPerClassSampler(
        train_dataset.targets, m=samples_per_class, batch_size=batch_size, length_before_new_iter=len(train_dataset)
    )
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

    return train_loader, val_dataset


def setup_training(model_checkpoint, num_classes, embedding_size, backbone, lr, epochs, train_loader):
    """Initialize model, optimizer, and scheduler."""
    logger.info("Initializing model and optimizer...")
    device = find_device()

    model = init_model(num_classes, embeddings=embedding_size, backbone_name=backbone,
                       checkpoint_path=model_checkpoint, device=device)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=epochs * len(train_loader))

    model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level="O2")
    amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    return model, optimizer, scheduler, loss_fn


def main():
    """Main function to run the classification training pipeline."""
    args = parse_args()
    device = find_device()

    train_records, val_records, label_to_id, id_to_label = prepare_dataset(args.dataset)

    output_folder = os.path.join(
        args.output_dir, args.dataset, "cross_entropy", datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )
    os.makedirs(output_folder, exist_ok=True)

    save_json(id_to_label, os.path.join(output_folder, "labels.json"))

    train_loader, val_dataset = prepare_dataloader(
        train_records, val_records, label_to_id, args.batch_size, args.classes_per_batch, args.samples_per_class
    )

    model, optimizer, scheduler, loss_fn = setup_training(
        args.checkpoint, val_dataset.n_classes, args.embedding_size, args.backbone, args.lr, args.epochs, train_loader
    )

    train(
        scheduler, args.epochs, optimizer, model, train_loader, val_dataset, device, ["accuracy"],
        loss_fn, logging, eval_every=5, file_name="model", output_folder=output_folder, extra_val=None
    )


if __name__ == "__main__":
    main()