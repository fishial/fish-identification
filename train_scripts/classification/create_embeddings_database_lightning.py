#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Embeddings Database for Lightning-trained Models.

This script creates an embeddings database from a trained Lightning model
that can be used with the interpreter for fast nearest-neighbor search.

Usage:
    python create_embeddings_database_lightning.py \
        --checkpoint checkpoints/best_model.ckpt \
        --dataset_name my_fiftyone_dataset \
        --output embeddings_database.pt \
        --config_json experiments/run_xyz/config.json \
        --labels_json experiments/run_xyz/labels.json \
        --val_tag val \
        --batch_size 64 \
        --device cuda
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Optional

import fiftyone as fo
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(0, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])

from module.classification_package.src.lightning_trainer_fixed import (
    ImageEmbeddingTrainerViT,
    ImageEmbeddingTrainerConvnext
)
from module.classification_package.src.datamodule import ImageEmbeddingDataModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    labels_mapping: Dict,
    device: str = 'cuda'
):
    """
    Load model from Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        config: Training configuration dictionary
        labels_mapping: Label to ID mapping
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    num_classes = len(labels_mapping)
    backbone_name = config['backbone_model_name']
    
    # Determine trainer class
    is_vit = 'convnext' not in backbone_name.lower()
    trainer_cls = ImageEmbeddingTrainerViT if is_vit else ImageEmbeddingTrainerConvnext
    
    logger.info(f"Loading {trainer_cls.__name__} from checkpoint...")
    
    # Load model
    model = trainer_cls.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        embedding_dim=config.get('embedding_dim', 512),
        backbone_model_name=backbone_name,
        arcface_s=config.get('arcface_s', 64.0),
        arcface_m=config.get('arcface_m', 0.2),
        pooling_type=config.get('pooling_type', 'attention'),
        map_location=device
    )
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model


def extract_embeddings(
    model,
    dataloader: DataLoader,
    device: str = 'cuda'
):
    """
    Extract embeddings from dataset using trained model.
    
    Args:
        model: Trained model
        dataloader: DataLoader for dataset
        device: Device to use for inference
    
    Returns:
        Dictionary with embeddings and metadata
    """
    all_embeddings = []
    all_labels = []
    all_image_ids = []
    all_annotation_ids = []
    all_drawn_fish_ids = []
    
    logger.info("Extracting embeddings...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            images, labels, object_masks = batch
            images = images.to(device)
            
            # Get embeddings
            embeddings, _, _ = model.model(
                images,
                return_softmax=False,
                return_attention_map=False
            )
            
            # Store results
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            # Get metadata if available
            if hasattr(dataloader.dataset, 'image_ids'):
                batch_size = len(labels)
                indices = range(len(all_image_ids), len(all_image_ids) + batch_size)
                all_image_ids.extend([dataloader.dataset.image_ids[i] for i in indices])
                all_annotation_ids.extend([dataloader.dataset.annotation_ids[i] for i in indices])
                all_drawn_fish_ids.extend([dataloader.dataset.drawn_fish_ids[i] for i in indices])
    
    # Concatenate embeddings
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    # If no metadata, create dummy values
    if not all_image_ids:
        all_image_ids = list(range(len(all_labels)))
        all_annotation_ids = list(range(len(all_labels)))
        all_drawn_fish_ids = list(range(len(all_labels)))
    
    logger.info(f"Extracted {len(embeddings_tensor)} embeddings")
    logger.info(f"Embedding shape: {embeddings_tensor.shape}")
    
    return {
        'embeddings': embeddings_tensor,
        'labels': all_labels,
        'image_id': all_image_ids,
        'annotation_id': all_annotation_ids,
        'drawn_fish_id': all_drawn_fish_ids,
    }


def create_labels_keys(labels_mapping: Dict, id_to_label: Dict) -> Dict:
    """
    Create labels_keys dictionary for database.
    
    Args:
        labels_mapping: Label name to internal ID mapping
        id_to_label: Internal ID to label name mapping
    
    Returns:
        Dictionary mapping internal_id -> {label, species_id}
    """
    labels_keys = {}
    
    for internal_id, label_name in id_to_label.items():
        labels_keys[int(internal_id)] = {
            'label': label_name,
            'species_id': int(internal_id)  # Using internal_id as species_id
        }
    
    return labels_keys


def main():
    parser = argparse.ArgumentParser(
        description="Create embeddings database from Lightning-trained model"
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to Lightning checkpoint (.ckpt file)'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='FiftyOne dataset name'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for embeddings database (.pt file)'
    )
    
    parser.add_argument(
        '--config_json',
        type=str,
        required=True,
        help='Path to training config.json'
    )
    
    parser.add_argument(
        '--labels_json',
        type=str,
        required=True,
        help='Path to labels.json from training'
    )
    
    parser.add_argument(
        '--val_tag',
        type=str,
        default='val',
        help='Dataset tag to use for creating embeddings'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    parser.add_argument(
        '--image_size',
        type=int,
        default=None,
        help='Image size (if None, will use from config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config_json}")
    with open(args.config_json, 'r') as f:
        config = json.load(f)
    
    # Load labels mapping
    logger.info(f"Loading labels from {args.labels_json}")
    with open(args.labels_json, 'r') as f:
        id_to_label = json.load(f)
    
    # Convert string keys to int
    id_to_label = {int(k): v for k, v in id_to_label.items()}
    label_to_id = {v: int(k) for k, v in id_to_label.items()}
    
    logger.info(f"Found {len(id_to_label)} classes")
    
    # Setup image size
    image_size = args.image_size or config.get('image_size', 224)
    logger.info(f"Using image size: {image_size}")
    
    # Create datamodule
    logger.info(f"Loading dataset: {args.dataset_name}")
    datamodule = ImageEmbeddingDataModule(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        classes_per_batch=32,  # Not used for validation
        samples_per_class=4,   # Not used for validation
        image_size=image_size,
        num_workers=args.num_workers,
        exclude_classes=None,
        train_tag=None,  # We only need validation data
        val_tag=args.val_tag,
        augmentation_preset='basic',
        use_cache=False,
    )
    
    datamodule.setup()
    
    # Override label_to_id to match training
    datamodule.label_to_id = label_to_id
    datamodule.num_classes = len(label_to_id)
    
    # Get validation dataloader
    val_loader = datamodule.val_dataloader()
    if val_loader is None:
        raise ValueError(f"No validation data found with tag '{args.val_tag}'")
    
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint,
        config,
        label_to_id,
        args.device
    )
    
    # Extract embeddings
    data = extract_embeddings(model, val_loader, args.device)
    
    # Add labels_keys
    data['labels_keys'] = create_labels_keys(label_to_id, id_to_label)
    
    # Save embeddings database
    logger.info(f"Saving embeddings database to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(data, args.output)
    
    # Verify saved file
    saved_data = torch.load(args.output)
    logger.info(f"✓ Embeddings database created successfully!")
    logger.info(f"  - Embeddings shape: {saved_data['embeddings'].shape}")
    logger.info(f"  - Number of samples: {len(saved_data['labels'])}")
    logger.info(f"  - Number of classes: {len(saved_data['labels_keys'])}")
    logger.info(f"  - File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
    
    # Print example usage
    print("\n" + "="*80)
    print("Embeddings database created successfully!")
    print("="*80)
    print("\nYou can now use it with the interpreter:")
    print("\nfrom interpreter_classifier_lightning import EmbeddingClassifier")
    print("\nconfig = {")
    print("    'log_level': 'INFO',")
    print(f"    'dataset': {{'path': '{args.output}'}},")
    print("    'model': {")
    print(f"        'checkpoint_path': '{args.checkpoint}',")
    print(f"        'backbone_model_name': '{config['backbone_model_name']}',")
    print(f"        'embedding_dim': {config['embedding_dim']},")
    print(f"        'num_classes': {len(id_to_label)},")
    print(f"        'arcface_s': {config['arcface_s']},")
    print(f"        'arcface_m': {config['arcface_m']},")
    print(f"        'pooling_type': '{config.get('pooling_type', 'attention')}',")
    print(f"        'device': '{args.device}'")
    print("    }")
    print("}")
    print("\nclassifier = EmbeddingClassifier(config)")
    print("="*80)


if __name__ == "__main__":
    main()
