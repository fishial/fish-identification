import sys
import os
import torch
import logging
import argparse
from datetime import datetime

import torchvision.models as models
from torchvision import transforms  # Added missing import
from apex import amp
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import samplers
from torch.utils.data.sampler import BatchSampler
import fiftyone as fo

# Import custom modules
# Modify sys.path to include the root directory containing 'fish-identification'
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
    print("SETUP: sys.path updated")
    
from module.classification_package.src.utils import (
    WarmupCosineSchedule, find_device, save_json, get_data_config
)
from module.classification_package.src.model import init_model
from module.classification_package.src.dataset import FishialDatasetFoOnlineCuting
from module.classification_package.src.train import train
from module.classification_package.src.loss_functions import *

# Initialize module-level logger
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments to configure training."""
    parser = argparse.ArgumentParser(
        description="Train an embedding network with command-line arguments (no YAML config)."
    )
    
    # Required arguments
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder for saving logs and models.")
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Name of the FiftyOne dataset for training.")
    parser.add_argument("--val_dataset", type=str, required=True,
                        help="Name of the FiftyOne dataset for validation.")
    parser.add_argument("--classes_per_batch", type=int, required=True,
                        help="Number of classes per batch for the batch sampler.")
    parser.add_argument("--samples_per_class", type=int, required=True,
                        help="Number of samples per class for the batch sampler.")
    parser.add_argument("--embeddings", type=int, required=True,
                        help="Dimension of the embeddings.")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Name of the model backbone.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a model checkpoint (optional).")
    parser.add_argument("--loss_name", type=str, choices=["quadruplet", "triplet", "tripletohnm", "angular", "pnploss"],
                        required=True, help="Loss function to use.")
    parser.add_argument("--adaptive_margin", type=float, default=None,
                        help="Adaptive margin value (if required by loss function).")
    parser.add_argument("--learning_rate", type=float, required=True,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--momentum", type=float, required=True,
                        help="Momentum for the optimizer.")
    parser.add_argument("--warmup_steps", type=int, required=True,
                        help="Number of warmup steps for the scheduler.")
    parser.add_argument("--epoch", type=int, required=True,
                        help="Number of training epochs.")
    parser.add_argument("--eval_epochs", type=int, default=1, help="Тumber of epochs to call validation")
    parser.add_argument("--opt_level", type=str, required=True,
                        help="Optimization level for apex (e.g., O1, O2).")
    
    # Optional arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda' or 'cpu').")
    
    return parser.parse_args()


def setup_logging(output_folder):
    """Setup logging to file and console."""
    log_file = os.path.join(output_folder, "training.log")
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    local_logger = logging.getLogger(__name__)
    local_logger.info(f"Logging initialized. Logs will be saved to {log_file}")
    return local_logger


def prepare_dataset(dataset_name, split):
    """Load FiftyOne dataset and split into train/validation."""
    logger.info(f"Loading FiftyOne dataset: {dataset_name} ({split})")
    fo_dataset = fo.load_dataset(dataset_name)

    if split == "train":
        return get_data_config(fo_dataset.match_tags("train"))
    elif split == "val":
        return get_data_config(fo_dataset.match_tags("val"))
    elif split == "":
        return get_data_config(fo_dataset)
    else:
        raise ValueError("Invalid dataset split. Use 'train', 'val', or ''.")


def get_train_transform():
    """Create and return the transformation for training images."""
    return transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.358, scale=(0.05, 0.4), ratio=(0.05, 6.1), value=0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """Create and return the transformation for validation images."""
    return transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def create_datasets(train_records, val_records, label_to_id, train_transform, val_transform):
    """Initialize and return the training and validation datasets."""
    train_dataset = FishialDatasetFoOnlineCuting(
        train_records, label_to_id, train_state=True,
        transform=train_transform, crop_type='rect'
    )
    val_dataset = FishialDatasetFoOnlineCuting(
        val_records, label_to_id,
        transform=val_transform, crop_type='rect'
    )
    return train_dataset, val_dataset

def create_train_loader(train_dataset, args):
    """Create and return a DataLoader for the training dataset using a balanced batch sampler."""
    batch_size = args.classes_per_batch * args.samples_per_class
    sampler = samplers.MPerClassSampler(
        train_dataset.targets,
        m=args.samples_per_class,
        batch_size=batch_size,
        length_before_new_iter=len(train_dataset)
    )
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=2,
        pin_memory=True
    )
    return train_loader

def select_loss_function(loss_name, adaptive_margin=None):
    """Select loss function based on provided argument."""
    loss_map = {
        "quadruplet": QuadrupletLoss(adaptive_margin),
        "triplet": TripletLoss(),
        "tripletohnm": WrapperOHNM(),
        "angular": WrapperAngular(),
        "pnploss": WrapperPNPLoss()
    }
    if loss_name not in loss_map:
        raise ValueError(f"Invalid loss function specified: {loss_name}")
    return loss_map[loss_name]


def train_model(args):
    """Main training function using command-line arguments."""
    global logger
    logger = setup_logging(args.output_folder)

    # Dataset preparation
    train_records = prepare_dataset(args.train_dataset, "train")
    val_records = prepare_dataset(args.train_dataset, "val")
    test_records = prepare_dataset(args.train_dataset, "val")
    
    label_to_id = {label: idx for idx, label in enumerate(list(train_records))}
    save_json(label_to_id, os.path.join(args.output_folder, "labels.json"))
    
    logger.info("Initializing datasets and DataLoaders...")
    
    # Create transformations for train and validation sets
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = FishialDatasetFoOnlineCuting(
        train_records, label_to_id, train_state=True,
        transform=train_transform, crop_type='rect'
    )
    
    val_dataset = FishialDatasetFoOnlineCuting(
        val_records, label_to_id,
        transform=val_transform, crop_type='rect'
    )
    
    test_dataset = FishialDatasetFoOnlineCuting(
        test_records, label_to_id,
        transform=val_transform, crop_type='rect'
    )
    
    # Create the DataLoader for training
    train_loader = create_train_loader(train_dataset, args)
    
    # Model setup
    device = args.device if args.device is not None else find_device()
    num_classes = len(train_records)
    model = init_model(num_classes, embeddings=args.embeddings,
                       backbone_name=args.backbone,
                       checkpoint_path=args.checkpoint, device=device)
    model.to(device)

    # Loss function
    loss_fn = select_loss_function(args.loss_name, args.adaptive_margin)

    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=0)
    scheduler = WarmupCosineSchedule(optimizer,
                                     warmup_steps=args.warmup_steps,
                                     t_total=args.epoch * len(train_loader))

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Start training
    train(scheduler, args.epoch, optimizer, model, train_loader, val_dataset,
          device, ["at_k"], loss_fn, logger,
          eval_every_epochs=2, output_folder=args.output_folder, test_set=test_dataset)


if __name__ == "__main__":
    args = parse_args()
    train_model(args)