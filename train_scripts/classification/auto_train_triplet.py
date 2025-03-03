import sys
import os
import yaml
import torch
import logging
import argparse
from datetime import datetime

import torchvision.models as models
from apex import amp
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import samplers
from torch.utils.data.sampler import BatchSampler
import fiftyone as fo

# Import custom modules
sys.path.insert(1, '')  # Modify path as needed
from module.classification_package.src.utils import (
    WarmupCosineSchedule, find_device, save_json, get_data_config
)
from module.classification_package.src.model import init_model
from module.classification_package.src.dataset import FishialDatasetFoOnlineCuting
from module.classification_package.src.train import train
from module.classification_package.src.loss_functions import *


# ============== CONFIGURATION ==============
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an embedding network.")

    # Required argument
    parser.add_argument("--config", "-c", required=True, help="Path to the config YAML file")
    
    # Optional overrides
    parser.add_argument("--device", type=str, default=None, help="Specify device (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    return parser.parse_args()


def load_config(path):
    """Load YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML config file: {exc}")


def setup_logging(output_folder):
    """Setup logging to file and console."""
    log_file = os.path.join(output_folder, "training.log")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Logs will be saved to {log_file}")
    return logger


def prepare_dataset(dataset_name, split):
    """Load FiftyOne dataset and split into train/validation."""
    logger.info(f"Loading FiftyOne dataset: {dataset_name} ({split})")
    fo_dataset = fo.load_dataset(dataset_name)

    if split == "train":
        return get_data_config(fo_dataset.match_tags("train"))
    elif split == "val":
        return get_data_config(fo_dataset.match_tags("val"))
    elif split == "full":
        return get_data_config(fo_dataset)
    else:
        raise ValueError("Invalid dataset split. Use 'train', 'val', or 'full'.")


def prepare_dataloader(train_records, val_records, label_to_id, config):
    """Prepare DataLoader with balanced batch sampling."""
    logger.info("Initializing datasets and DataLoaders...")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.358, scale=(0.05, 0.4), ratio=(0.05, 6.1), value=0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = FishialDatasetFoOnlineCuting(train_records, label_to_id, train_state=True, transform=transform_train, crop_type='rect')
    val_dataset = FishialDatasetFoOnlineCuting(val_records, label_to_id, transform=transform_val, crop_type='rect')

    batch_size = config["dataset"]["batchsampler"]["classes_per_batch"] * config["dataset"]["batchsampler"]["samples_per_class"]
    sampler = samplers.MPerClassSampler(train_dataset.targets, m=config["dataset"]["batchsampler"]["samples_per_class"], batch_size=batch_size, length_before_new_iter=len(train_dataset))
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=2, pin_memory=True)

    return train_loader, val_dataset


def select_loss_function(loss_name, adaptive_margin=None):
    """Select loss function based on config."""
    loss_map = {
        "quadruplet": QuadrupletLoss(adaptive_margin),
        "triplet": TripletLoss(),
        "tripletohnm": WrapperOHNM(),
        "angular": WrapperAngular(),
        "pnploss": WrapperPNPLoss()
    }
    return loss_map.get(loss_name, None)


def train_model(config):
    """Main training function."""
    global logger

    # Output directory setup
    output_folder = config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    logger = setup_logging(output_folder)

    # Dataset preparation
    train_records = prepare_dataset(config["dataset"]["train"], "train")
    val_records = prepare_dataset(config["dataset"]["val"], "val")
    label_to_id = {label: idx for idx, label in enumerate(list(train_records))}
    save_json(label_to_id, os.path.join(output_folder, "labels.json"))

    train_loader, val_dataset = prepare_dataloader(train_records, val_records, label_to_id, config)

    # Model setup
    device = config["device"] or find_device()
    model = init_model(len(train_records), embeddings=config["model"]["embeddings"], backbone_name=config["model"]["backbone"], checkpoint_path=config["checkpoint"], device=device)
    model.to(device)

    # Loss function
    loss_fn = select_loss_function(config["train"]["loss"]["name"], config["train"]["loss"].get("adaptive_margin"))
    if loss_fn is None:
        raise ValueError("Invalid loss function specified in config.")

    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=config["train"]["learning_rate"], momentum=config["train"]["momentum"], weight_decay=0)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=config["train"]["warmup_steps"], t_total=config["train"]["epoch"] * len(train_loader))

    model, optimizer = amp.initialize(model=model, optimizers=optimizer, opt_level=config["train"]["opt_level"])
    amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Train
    train(scheduler, config["train"]["epoch"], optimizer, model, train_loader, val_dataset, device, ["at_k"], loss_fn, logger, eval_every=20, file_name=config["file_name"], output_folder=output_folder)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config["output_folder"] = args.output_dir

    train_model(config)