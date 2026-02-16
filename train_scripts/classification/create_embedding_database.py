#!/usr/bin/env python3
"""
Create Embedding Database for k-NN Classification

Usage:
    python create_embedding_database.py \
        --checkpoint /path/to/model.ckpt \
        --dataset_name classification_v0.10_train \
        --output_dir /path/to/output \
        --samples_per_class 100
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter

import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import pairwise_distances
import faiss
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from module.classification_package.src.lightning_trainer_fixed import ImageEmbeddingTrainerViT
from module.classification_package.src.datamodule import ImageEmbeddingDataModule

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_sequential_dataloader(dataset, batch_size, num_workers=4):
    """Create a sequential (non-shuffled) DataLoader for embedding extraction."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: sequential order
        num_workers=num_workers,
        pin_memory=True
    )


def extract_embeddings(model, dataloader, device):
    """Extract normalized embeddings from model."""
    embeddings = []
    labels = []
    logits_list = []
    image_ids = []
    annotation_ids = []
    drawn_fish_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            images, targets, metadata = batch
            images = images.to(device)
            
            # Get embeddings and logits
            emb_norm, logits, _ = model(images)
            
            embeddings.append(emb_norm.cpu())
            logits_list.append(logits.cpu())
            labels.extend(targets.cpu().numpy())
            
            # Metadata
            if 'image_id' in metadata:
                image_ids.extend(metadata['image_id'])
            if 'annotation_id' in metadata:
                annotation_ids.extend(metadata['annotation_id'])
            if 'drawn_fish_id' in metadata:
                drawn_fish_ids.extend(metadata['drawn_fish_id'])
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    logits = torch.cat(logits_list, dim=0).numpy()
    labels = np.array(labels)
    
    return {
        'embeddings': embeddings,
        'logits': logits,
        'labels': labels,
        'image_ids': image_ids if image_ids else None,
        'annotation_ids': annotation_ids if annotation_ids else None,
        'drawn_fish_ids': drawn_fish_ids if drawn_fish_ids else None,
    }


def compute_centroids(embeddings, labels):
    """Compute normalized centroid for each class."""
    unique_labels = np.unique(labels)
    centroids = {}
    
    for label in tqdm(unique_labels, desc='Computing centroids'):
        class_embeddings = embeddings[labels == label]
        centroid = np.mean(class_embeddings, axis=0)
        # Normalize
        centroid /= (np.linalg.norm(centroid) + 1e-10)
        centroids[label] = centroid
    
    return centroids


def select_representative_samples(embeddings, labels, metadata, centroids, samples_per_class=100):
    """Select top-N most representative samples per class (closest to centroid)."""
    selected_indices = []
    unique_labels = np.unique(labels)
    
    for label in tqdm(unique_labels, desc='Selecting samples'):
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # Compute distances to centroid
        centroid = centroids[label]
        distances = 1.0 - np.dot(class_embeddings, centroid)  # Cosine distance
        
        # Select top-N closest to centroid
        n_select = min(samples_per_class, len(class_indices))
        top_n_local_indices = np.argsort(distances)[:n_select]
        top_n_global_indices = class_indices[top_n_local_indices]
        
        selected_indices.extend(top_n_global_indices)
    
    selected_indices = np.array(selected_indices)
    
    # Create filtered database
    db = {
        'embeddings': embeddings[selected_indices],
        'labels': labels[selected_indices],
    }
    
    # Add metadata if available
    for key in ['image_ids', 'annotation_ids', 'drawn_fish_ids']:
        if key in metadata and metadata[key] is not None:
            db[key] = [metadata[key][i] for i in selected_indices]
    
    return db, selected_indices


def main(args):
    logger.info("Starting embedding database creation...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = ImageEmbeddingTrainerViT.load_from_checkpoint(
        args.checkpoint,
        map_location=device
    ).eval()
    logger.info(f"Model loaded: embedding_dim={model.hparams.embedding_dim}, num_classes={model.hparams.num_classes}")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    datamodule = ImageEmbeddingDataModule(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        classes_per_batch=32,  # Not used for inference, but required
        samples_per_class=6,   # Not used for inference, but required
        image_size=224,
        exclude_classes=args.exclude_classes.split(',') if args.exclude_classes else [],
        augmentation_preset='none',  # No augmentation for embedding extraction
        train_tag=args.train_tag,
        val_tag=args.val_tag,
        num_workers=args.num_workers,
    )
    datamodule.setup('fit')
    logger.info(f"Dataset: train={len(datamodule.train_dataset)}, val={len(datamodule.val_dataset)}, classes={len(datamodule.labels_keys)}")
    
    # Extract train embeddings
    # Note: Use sequential dataloader (not MPerClassSampler) to preserve order
    logger.info("Extracting train embeddings...")
    train_dataloader = create_sequential_dataloader(
        datamodule.train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_data = extract_embeddings(model, train_dataloader, device)
    logger.info(f"Extracted {len(train_data['embeddings'])} train embeddings")
    
    # Compute centroids
    logger.info("Computing class centroids...")
    centroids = compute_centroids(train_data['embeddings'], train_data['labels'])
    logger.info(f"Computed {len(centroids)} class centroids")
    
    # Select representative samples
    logger.info(f"Selecting top-{args.samples_per_class} samples per class...")
    database, selected_indices = select_representative_samples(
        train_data['embeddings'],
        train_data['labels'],
        {
            'image_ids': train_data['image_ids'],
            'annotation_ids': train_data['annotation_ids'],
            'drawn_fish_ids': train_data['drawn_fish_ids'],
        },
        centroids,
        samples_per_class=args.samples_per_class
    )
    logger.info(f"Selected {len(database['embeddings'])} samples for database")
    
    # Statistics
    label_counts = Counter(database['labels'])
    logger.info(f"Samples per class: min={min(label_counts.values())}, max={max(label_counts.values())}, mean={np.mean(list(label_counts.values())):.1f}")
    
    # Save database
    database_path = output_dir / f'embedding_database_top{args.samples_per_class}.pt'
    torch.save({
        'embeddings': torch.from_numpy(database['embeddings']),
        'labels': database['labels'],
        'image_ids': database.get('image_ids'),
        'annotation_ids': database.get('annotation_ids'),
        'drawn_fish_ids': database.get('drawn_fish_ids'),
        'labels_keys': datamodule.labels_keys,
        'centroids': centroids,
        'config': vars(args),
    }, database_path)
    logger.info(f"✅ Database saved to: {database_path}")
    logger.info(f"   Size: {database_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save metadata
    metadata = {
        'checkpoint': args.checkpoint,
        'dataset_name': args.dataset_name,
        'num_samples': len(database['embeddings']),
        'num_classes': len(centroids),
        'samples_per_class': args.samples_per_class,
        'embedding_dim': train_data['embeddings'].shape[1],
        'database_path': str(database_path),
    }
    metadata_path = output_dir / 'database_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Metadata saved to: {metadata_path}")
    
    logger.info("Done! ✅")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create embedding database for k-NN classification')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='FiftyOne dataset name')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for database')
    
    # Optional
    parser.add_argument('--samples_per_class', type=int, default=100,
                        help='Number of samples per class (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--exclude_classes', type=str, default='unset,Thunnus obesus',
                        help='Comma-separated classes to exclude')
    parser.add_argument('--train_tag', type=str, default='train',
                        help='Train tag (default: train)')
    parser.add_argument('--val_tag', type=str, default='val',
                        help='Validation tag (default: val)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU (default: use CUDA if available)')
    
    args = parser.parse_args()
    main(args)
