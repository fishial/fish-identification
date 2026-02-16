# -*- coding: utf-8 -*-
"""
Data Module for Fish Classification Training.

This module provides:
- FiftyOne dataset integration
- Configurable data augmentation pipelines
- Balanced batch sampling for metric learning
- Support for different augmentation presets

Augmentation Presets:
- 'basic': Minimal augmentations (resize, normalize)
- 'standard': Standard augmentations (flips, color jitter, dropout)
- 'medium': Moderate augmentations (between standard and strong)
- 'strong': Aggressive augmentations for underwater fish images
"""

import os
import sys
import logging
import json
import hashlib
from typing import Optional, Literal

from tqdm import tqdm

import albumentations as A
import fiftyone as fo
from albumentations.pytorch import ToTensorV2
from lightning.pytorch import LightningDataModule
from pytorch_metric_learning import samplers
from torch.utils.data import BatchSampler, DataLoader

from module.classification_package.src.dataset import FishialDatasetFoOnlineCuting

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ImageEmbeddingDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for fish image embedding training.
    
    Features:
    - FiftyOne dataset integration
    - Balanced batch sampling (MPerClassSampler)
    - Configurable augmentation presets
    - Class exclusion support
    - Caching for faster repeated runs
    
    Args:
        dataset_name: Name of the FiftyOne dataset
        batch_size: Batch size for validation loader
        classes_per_batch: Number of classes per batch for MPerClassSampler
        samples_per_class: Number of samples per class in each batch
        image_size: Input image size (H=W)
        num_workers: Number of data loading workers
        exclude_classes: List of class labels to exclude
        train_tag: Dataset tag for training samples
        val_tag: Dataset tag for validation samples
        augmentation_preset: One of 'basic', 'standard', 'medium', 'strong'
        cache_dir: Directory to store cache files (defaults to parent of output_dir)
        use_cache: Whether to use caching (default: True)
        class_mapping_path: Optional path to a JSON file mapping class IDs to names
    """
    
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        classes_per_batch: int,
        samples_per_class: int,
        image_size: int,
        num_workers: int,
        exclude_classes: Optional[list] = None,
        train_tag: Optional[str] = "train",
        val_tag: Optional[str] = "val",
        augmentation_preset: Literal['basic', 'standard', 'medium', 'strong'] = 'standard',
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        class_mapping_path: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.class_mapping_path = class_mapping_path
        self._external_label_mapping = self._load_external_label_mapping(class_mapping_path)
        self.train_dataset = None
        self.val_dataset = None
        self.label_to_id = {}
        self.num_classes = 0
        
    def _get_cache_key(self, dataset_name: str, train_tag: Optional[str], val_tag: Optional[str]) -> str:
        """
        Generate a unique cache key based on dataset configuration.
        
        Args:
            dataset_name: FiftyOne dataset name
            train_tag: Training tag
            val_tag: Validation tag
            
        Returns:
            MD5 hash as cache key
        """
        config_str = f"{dataset_name}_{train_tag}_{val_tag}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_cache_path(self, dataset_name: str, train_tag: Optional[str], val_tag: Optional[str]) -> str:
        """
        Get the path to the cache file.
        
        Args:
            dataset_name: FiftyOne dataset name
            train_tag: Training tag
            val_tag: Validation tag
            
        Returns:
            Path to cache file
        """
        cache_dir = self.hparams.cache_dir
        if cache_dir is None:
            # Default: use a .cache directory in user's home
            cache_dir = os.path.expanduser("~/.cache/fish_identification")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_key = self._get_cache_key(dataset_name, train_tag, val_tag)
        cache_filename = f"dataset_cache_{cache_key}.json"
        return os.path.join(cache_dir, cache_filename)
    
    def _save_cache(self, cache_path: str, train_records: dict, val_records: dict):
        """
        Save processed data to cache file.
        
        Args:
            cache_path: Path to save cache
            train_records: Training data records
            val_records: Validation data records
        """
        cache_data = {
            "train_records": train_records,
            "val_records": val_records,
            "dataset_name": self.hparams.dataset_name,
            "train_tag": self.hparams.train_tag,
            "val_tag": self.hparams.val_tag,
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache saved to: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self, cache_path: str) -> Optional[tuple[dict, dict]]:
        """
        Load processed data from cache file.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            Tuple of (train_records, val_records) or None if cache invalid
        """
        if not os.path.exists(cache_path):
            logger.info("No cache found, will process dataset from scratch")
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache
            if (cache_data.get("dataset_name") != self.hparams.dataset_name or
                cache_data.get("train_tag") != self.hparams.train_tag or
                cache_data.get("val_tag") != self.hparams.val_tag):
                logger.warning("Cache configuration mismatch, will regenerate")
                return None
            
            train_records = cache_data.get("train_records", {})
            val_records = cache_data.get("val_records", {})
            
            logger.info(f"Loaded cache from: {cache_path}")
            logger.info(f"Train classes: {len(train_records)}, Val classes: {len(val_records)}")
            
            return train_records, val_records
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will regenerate")
            return None
        
    def _get_data_config(self, dataset):
        """Extract data configuration from FiftyOne dataset."""
        labels_dict = {}
        for sample in tqdm(dataset, desc="Loading dataset"):
            base_name = os.path.basename(sample['filepath'])
            width = sample['width']
            height = sample['height']

            polyline = sample['polyline']
            
            if polyline['label'] not in labels_dict:
                labels_dict.update({polyline['label']: []})

            poly = [[int(point[0] * width), int(point[1] * height)] for point in polyline['points'][0]]
            labels_dict[polyline['label']].append({
                'id': sample['annotation_id'],
                'name': polyline['label'],
                'base_name': base_name,
                'poly': poly,
                'file_name': sample['filepath'],
                'drawn_fish_id': sample['drawn_fish_id'],
                'annotation_id': sample['annotation_id'],
                'image_id': sample['image_id']
            })
        return labels_dict
    
    def _get_data_config_combined(self, fo_dataset, train_tag: Optional[str], val_tag: Optional[str]) -> tuple[dict, dict]:
        """
        Extract data configuration for both train and val in a single pass.
        
        Args:
            fo_dataset: FiftyOne dataset
            train_tag: Tag for training samples
            val_tag: Tag for validation samples
            
        Returns:
            Tuple of (train_records, val_records)
        """
        train_records = {}
        val_records = {}
        
        logger.info("Processing dataset (single pass for train and val)...")
        
        for sample in tqdm(fo_dataset, desc="Loading dataset"):
            base_name = os.path.basename(sample['filepath'])
            width = sample['width']
            height = sample['height']
            polyline = sample['polyline']
            
            poly = [[int(point[0] * width), int(point[1] * height)] for point in polyline['points'][0]]
            
            record = {
                'id': sample['annotation_id'],
                'name': polyline['label'],
                'base_name': base_name,
                'image_id': sample['image_id'],
                'poly': poly,
                'file_name': sample['filepath'],
                'drawn_fish_id': sample['drawn_fish_id'],
                'annotation_id': sample['annotation_id']
            }
            
            # Determine which split(s) this sample belongs to
            # Use hasattr to safely check if tags exist, then access as attribute
            sample_tags = set(sample.tags) if hasattr(sample, 'tags') and sample.tags else set()
            
            # Add to train records
            if train_tag is None or train_tag in sample_tags:
                if polyline['label'] not in train_records:
                    train_records[polyline['label']] = []
                train_records[polyline['label']].append(record)
            
            # Add to val records
            if val_tag and val_tag in sample_tags:
                if polyline['label'] not in val_records:
                    val_records[polyline['label']] = []
                val_records[polyline['label']].append(record)
        
        logger.info(f"Processed train classes: {len(train_records)}, val classes: {len(val_records)}")
        return train_records, val_records

    def _load_external_label_mapping(self, class_mapping_path: Optional[str]) -> Optional[dict[str, int]]:
        """
        Load a custom label mapping from a JSON file.
        """
        if not class_mapping_path:
            return None

        expanded_path = os.path.abspath(os.path.expanduser(class_mapping_path))
        if not os.path.isfile(expanded_path):
            raise FileNotFoundError(f"Class mapping file not found: {expanded_path}")

        try:
            with open(expanded_path, 'r', encoding='utf-8') as mapping_file:
                raw_mapping = json.load(mapping_file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse class mapping JSON at {expanded_path}: {exc}") from exc

        if not isinstance(raw_mapping, dict):
            raise ValueError("Class mapping must be a JSON object with id -> label pairs.")

        parsed_mapping: dict[str, int] = {}
        seen_ids: set[int] = set()
        for raw_id, label_value in raw_mapping.items():
            if label_value is None:
                continue
            label_name = str(label_value).strip()
            if not label_name:
                continue
            try:
                label_id = int(raw_id)
            except (TypeError, ValueError):
                logger.warning("Skipping class mapping entry with non-integer id: %s", raw_id)
                continue
            if label_id in seen_ids:
                logger.warning("Duplicate label id %s in mapping, skipping duplicate entry.", label_id)
                continue
            if label_name in parsed_mapping:
                logger.warning("Label '%s' already mapped, skipping duplicate entry.", label_name)
                continue
            seen_ids.add(label_id)
            parsed_mapping[label_name] = label_id

        if not parsed_mapping:
            raise ValueError("No valid class mappings found in the provided JSON file.")

        logger.info("Loaded %d classes from mapping file: %s", len(parsed_mapping), expanded_path)
        return parsed_mapping

    def _apply_external_label_mapping(self, train_records: dict, val_records: dict) -> tuple[dict, dict]:
        """
        Filter dataset records to only include classes defined in the custom mapping.
        """
        mapping = self._external_label_mapping
        if not mapping:
            return train_records, val_records

        allowed_labels = set(mapping.keys())
        filtered_train = {label: samples for label, samples in train_records.items() if label in allowed_labels}
        filtered_val = {label: samples for label, samples in val_records.items() if label in allowed_labels}

        missing_labels = allowed_labels - set(filtered_train)
        if missing_labels:
            sample = ", ".join(sorted(list(missing_labels))[:5])
            suffix = "..." if len(missing_labels) > 5 else ""
            logger.warning(
                "Custom mapping contains %d classes not present in the dataset: %s%s",
                len(missing_labels),
                sample,
                suffix,
            )

        dropped_labels = set(train_records) - set(filtered_train)
        if dropped_labels:
            sample = ", ".join(sorted(list(dropped_labels))[:5])
            suffix = "..." if len(dropped_labels) > 5 else ""
            logger.info(
                "Dropping %d dataset classes not in mapping: %s%s",
                len(dropped_labels),
                sample,
                suffix,
            )

        if not filtered_train:
            raise ValueError("Applying the custom label mapping removed all training classes.")

        return filtered_train, filtered_val

    def _build_label_to_id(self, train_records: dict) -> dict[str, int]:
        """
        Build the label -> id mapping used by the dataset.
        """
        if not train_records:
            raise ValueError("No training records available to build label mapping.")

        if self._external_label_mapping:
            available_labels = [label for label in train_records if label in self._external_label_mapping]
            if not available_labels:
                raise ValueError("Custom label mapping does not cover any of the retained training classes.")
            sorted_labels = sorted(available_labels, key=lambda label: self._external_label_mapping[label])
            return {label: self._external_label_mapping[label] for label in sorted_labels}

        return {label: idx for idx, label in enumerate(train_records)}

    def get_transform(self, is_train: bool):
        """
        Get data augmentation pipeline based on preset.
        
        Args:
            is_train: Whether this is for training (True) or validation (False)
        
        Returns:
            Albumentations Compose object
        """
        preset = self.hparams.augmentation_preset
        image_size = self.hparams.image_size
        
        if not is_train:
            # Validation: minimal transforms
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        if preset == 'basic':
            return self._get_basic_transform(image_size)
        elif preset == 'standard':
            return self._get_standard_transform(image_size)
        elif preset == 'medium':
            return self._get_medium_transform(image_size)
        elif preset == 'strong':
            return self._get_strong_transform(image_size)
        else:
            logger.warning(f"Unknown augmentation preset: {preset}, using 'standard'")
            return self._get_standard_transform(image_size)

    def _get_basic_transform(self, image_size: int):
        """Basic transform: minimal augmentations."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _get_standard_transform(self, image_size: int):
        """Standard transform: balanced augmentations."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Geometric augmentations
            A.Affine(
                translate_percent=0.05, 
                scale=(0.9, 1.1), 
                rotate=(-15, 15), 
                p=0.5
            ),
            
            # Color/Noise augmentations
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.HueSaturationValue(p=1.0),
                A.RGBShift(p=1.0)
            ], p=0.5),
            
            # Regularization (Cutout equivalent)
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(image_size // 20, image_size // 10),
                hole_width_range=(image_size // 20, image_size // 10),
                fill=0,
                p=0.2,
            ),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _get_medium_transform(self, image_size: int):
        """
        Medium transform: moderate augmentations between standard and strong.
        
        Optimized for underwater fish images with balanced regularization.
        Includes:
        - Moderate geometric transforms
        - Underwater-specific color adjustments
        - Light blur simulation
        - Moderate regularization
        """
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # Basic flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            
            # Moderate geometric transforms - various viewing angles
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,
                p=0.5
            ),
            
            # Underwater-specific color augmentations
            A.OneOf([
                # Standard color jitter
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=1.0
                ),
                # Underwater color shifts (blue/green tints)
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                # Color jitter for lighting variations
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.3,
                    hue=0.1,
                    p=1.0
                ),
                # RGB shift for underwater color absorption
                A.RGBShift(
                    r_shift_limit=15,
                    g_shift_limit=20,
                    b_shift_limit=25,
                    p=1.0
                ),
            ], p=0.6),
            
            # Light underwater blur/quality simulation
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Moderate regularization
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(image_size // 20, image_size // 12),
                hole_width_range=(image_size // 20, image_size // 12),
                fill=0,
                p=0.3,
            ),
            
            # Final normalization
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _get_strong_transform(self, image_size: int):
        """
        Strong transform: aggressive augmentations optimized for underwater fish images.
        
        Includes:
        - Enhanced geometric transforms for different viewing angles
        - Underwater-specific color shifts
        - Blur/quality degradation simulation
        - Strong regularization
        """
        return A.Compose([
            A.Resize(image_size, image_size),
            
            # Basic flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # Fish often photographed from above
            
            # Strong geometric transforms - various viewing angles
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.25,
                rotate_limit=45,
                border_mode=0,  # Constant border
                p=0.6
            ),
            
            # Perspective distortion - simulates different camera angles
            A.Perspective(
                scale=(0.02, 0.08),
                p=0.3
            ),
            
            # Underwater-specific color augmentations
            A.OneOf([
                # Standard color jitter
                A.RandomBrightnessContrast(
                    brightness_limit=0.35,
                    contrast_limit=0.35,
                    p=1.0
                ),
                # Underwater color shifts (blue/green tints)
                A.HueSaturationValue(
                    hue_shift_limit=25,  # Underwater often has color casts
                    sat_shift_limit=40,
                    val_shift_limit=30,
                    p=1.0
                ),
                # Color jitter for lighting variations
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.4,
                    hue=0.15,
                    p=1.0
                ),
                # RGB shift for underwater color absorption
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=25,
                    b_shift_limit=30,  # More blue shift underwater
                    p=1.0
                ),
            ], p=0.8),
            
            # Additional underwater effects
            A.OneOf([
                # Green/Blue tint common in underwater photos
                A.ToSepia(p=0.3),  # Slight sepia can simulate murky water
                A.ChannelShuffle(p=0.1),  # Extreme color variation
            ], p=0.15),
            
            # Underwater blur/quality simulation
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=(3, 9), p=1.0),  # Moving fish/camera
                A.MedianBlur(blur_limit=5, p=1.0),
                A.Defocus(radius=(3, 7), p=1.0),  # Out of focus
            ], p=0.35),
            
            # Noise - simulates underwater camera noise
            A.OneOf([
                A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.25),
            
            # Simulate underwater lighting conditions
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    p=1.0
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_range=(0, 1),
                    num_flare_circles_range=(1, 3),
                    src_radius=100,
                    p=0.5
                ),
            ], p=0.2),
            
            # Strong regularization
            A.CoarseDropout(
                num_holes_range=(1, 10),
                hole_height_range=(image_size // 16, image_size // 8),
                hole_width_range=(image_size // 16, image_size // 8),
                fill=0,
                p=0.4,
            ),
            
            # GridDropout - alternative regularization
            A.GridDropout(
                ratio=0.3,
                unit_size_range=(image_size // 8, image_size // 4),
                random_offset=True,
                p=0.2
            ),
            
            # Final normalization
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _filter_labels(self, records: dict, exclude_set: set) -> dict:
        """Filter out excluded class labels."""
        if not exclude_set:
            return records
        return {label: items for label, items in records.items() if label not in exclude_set}

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets and label mappings.
        
        Called on every GPU to load datasets and create mappings.
        Uses caching to avoid reprocessing the same dataset configuration.
        """
        logger.info(f"Setting up data for stage: {stage}, dataset: {self.hparams.dataset_name}")
        
        try:
            fo_dataset = fo.load_dataset(self.hparams.dataset_name)
        except ValueError:
            logger.warning(f"Dataset '{self.hparams.dataset_name}' not found. Creating a dummy dataset.")

        train_tag = self.hparams.train_tag
        val_tag = self.hparams.val_tag
        
        # Try to load from cache first
        train_records = None
        val_records = None
        
        if self.hparams.use_cache:
            cache_path = self._get_cache_path(self.hparams.dataset_name, train_tag, val_tag)
            cached_data = self._load_cache(cache_path)
            
            if cached_data is not None:
                train_records, val_records = cached_data
        
        # If no cache or cache disabled, process dataset
        if train_records is None:
            logger.info("Processing dataset from FiftyOne...")
            train_records, val_records = self._get_data_config_combined(fo_dataset, train_tag, val_tag)
            
            # Save to cache for future runs
            if self.hparams.use_cache:
                cache_path = self._get_cache_path(self.hparams.dataset_name, train_tag, val_tag)
                self._save_cache(cache_path, train_records, val_records)

        if self._external_label_mapping:
            train_records, val_records = self._apply_external_label_mapping(train_records, val_records)

        exclude_set = set(self.hparams.exclude_classes or [])
        if exclude_set:
            logger.info("Excluding classes from train/val: %s", sorted(exclude_set))
            train_records = self._filter_labels(train_records, exclude_set)
            val_records = self._filter_labels(val_records, exclude_set)
        
        self.label_to_id = self._build_label_to_id(train_records)
        self.num_classes = len(self.label_to_id)
        
        self.train_dataset = FishialDatasetFoOnlineCuting(
            train_records, self.label_to_id, transform=self.get_transform(is_train=True)
        )
        if val_records:
            self.val_dataset = FishialDatasetFoOnlineCuting(
                val_records, self.label_to_id, transform=self.get_transform(is_train=False)
            )
            if len(self.val_dataset) == 0:
                logger.warning("Validation tag '%s' produced 0 samples; disabling validation.", val_tag)
                self.val_dataset = None
        else:
            logger.warning("Validation tag '%s' produced no records; disabling validation.", val_tag)
            self.val_dataset = None
            
        logger.info(f"Setup complete. Found {self.num_classes} classes.")
        logger.info(f"Augmentation preset: {self.hparams.augmentation_preset}")
        if self.val_dataset is None:
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Val dataset: disabled")
        else:
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        """Create training dataloader with balanced sampler."""
        effective_batch_size = self.hparams.classes_per_batch * self.hparams.samples_per_class
        
        m_per_class_sampler = samplers.MPerClassSampler(
            labels=self.train_dataset.targets,
            m=self.hparams.samples_per_class,
            batch_size=effective_batch_size,
            length_before_new_iter=len(self.train_dataset)
        )
        
        batch_sampler = BatchSampler(
            m_per_class_sampler, 
            batch_size=effective_batch_size, 
            drop_last=False
        )
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
