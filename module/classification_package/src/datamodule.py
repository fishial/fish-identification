# -*- coding: utf-8 -*-
import os
import logging
import json
import hashlib
from typing import Optional, Literal

import cv2
import torch
import albumentations as A
import fiftyone as fo
from albumentations.pytorch import ToTensorV2
from lightning.pytorch import LightningDataModule
from pytorch_metric_learning import samplers
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# 🚨 FIX 1: Import the correct name of the new dataset
from module.classification_package.src.dataset import FishialDatasetOnlineCutting
from module.classification_package.src.custom_sampler import BalancedBatchSampler

logger = logging.getLogger(__name__)

class ImageEmbeddingDataModule(LightningDataModule):
    """
    DataModule for fish embedding training.
    Optimized for DINOv2 (vit_base_patch14_reg4_dinov2)
    """

    def __init__(
        self,
        dataset_name: str,
        image_size: tuple,
        class_mapping_path: str,
        classes_per_batch: int = 1,
        samples_per_class: int = 1,
        train_tags: list[str] = None,
        val_tags: list[str] = None,
        labels_path: str = None,
        num_workers: int = 8,
        val_batch_size: int = 32,
        val_num_workers: int = 2,
        augmentation_preset: Literal['basic', 'standard', 'medium', 'strong'] = 'basic',
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        instance_data: bool = False,
        bg_removal_prob: float = 0.0,
        bbox_padding_limit: float = 0.0,
        resize_strategy : Literal['pad', 'squish'] = 'pad',
        alignment_method : Literal['diagonal', 'horizontal'] = 'horizontal',
        prefetch_factor: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_dataset = None
        self.val_dataset = None

        # 1. First, load class_mapping (needed as a fallback if labels_path == None)
        self.class_mapping = {}
        if class_mapping_path and os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
        else:
            logger.warning(f"Class mapping file not found at {class_mapping_path}!")

        # 2. Define the list of classes for training
        if labels_path is not None and os.path.exists(labels_path):
            # If path is provided and file exists — read from it
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels_to_train = set(line.strip() for line in f)
        else:
            # If labels_path == None (or file missing), take all labels from class_mapping
            if labels_path is not None:
                logger.warning(f"Labels file not found at {labels_path}. Falling back to all labels in class_mapping.")
            
            self.labels_to_train = set(self.class_mapping.keys())

        # 3. Form the labels_keys dictionary
        self.labels_keys = {}
        for label_name in self.labels_to_train:
            if label_name in self.class_mapping:
                label_data = self.class_mapping[label_name]
                if label_data['id'] is None: continue

                # Use .get() for fishial_extra to avoid KeyError if JSON structure is incomplete
                species_id = label_data.get('fishial_extra', {}).get('species_id', None)
                
                self.labels_keys[int(label_data['id'])] = {
                    "label": label_name, 
                    "species_id": species_id
                }
            else:
                logger.warning(f"Label '{label_name}' from labels_to_train was not found in class_mapping!")

        self.num_classes = len(self.class_mapping)

    # --- Caching ---
    def _get_cache_path(self) -> str:
        cache_dir = self.hparams.cache_dir or os.path.expanduser("~/.cache/fishial_data")
        os.makedirs(cache_dir, exist_ok=True)
        # 💡 CHANGE: Cache now depends ONLY on the dataset name (FiftyOne).
        # Changing tags or labels_to_train no longer invalidates the cache!
        config_hash = hashlib.md5(self.hparams.dataset_name.encode()).hexdigest()
        return os.path.join(cache_dir, f"dataset_cache_all_records_{config_hash}.json")

    # --- Augmentations ---
    def get_transform(self, is_train: bool) -> A.Compose:
        img_h, img_w = self.hparams.image_size # (154, 420)
        preset = self.hparams.augmentation_preset
        NEUTRAL_BG = (124, 116, 104) # Ensure this matches RGB/BGR order

        # 1. Resize (now works perfectly thanks to the gray canvas approach)
        resize_ops = [
            A.Resize(height=img_h, width=img_w, interpolation=cv2.INTER_LINEAR)
        ]
        
        # 2. Base normalization
        base_norm_tensor = [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        if not is_train or preset == 'basic':
            return A.Compose(resize_ops + base_norm_tensor)

        # 3. Basic safe augmentations
        train_pipeline = resize_ops + [
            # Horizontal flip is usually OK for fish (unless you have specific asymmetrical species)
            A.HorizontalFlip(p=0.5), 
            # ROTATION REMOVED! It is now handled inside _crop_horizontal
        ]

        if preset in ('medium', 'strong'):
            # --- Block 1: Water Column Distortions (Optics) ---
            water_optics = A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), # Simulates uneven underwater lighting
            ], p=0.6)
            
            # --- Block 2: Underwater Filming Defects (Noise/Blur) ---
            camera_noise = A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0), # Fast swimming fish
                A.GaussianBlur(blur_limit=(3, 5), p=1.0), # Defocus
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0), # Poor camera sensor
                A.ImageCompression(quality_range=(60, 95), p=1.0), # Compression artifacts (WhatsApp/Telegram)
            ], p=0.5)

            train_pipeline.extend([water_optics, camera_noise])

            if preset == 'strong':
                # --- Block 3: SAFE CoarseDropout (Simulating Occlusions) ---
                # We make holes small enough to cover parts of fins/tails 
                # but not kill the entire fish visualization.
                occlusion_op = A.OneOf([
                    # 1. Vertical occlusion (seaweed, fishing line)
                    A.CoarseDropout(
                        num_holes_range=(1, 2),
                        hole_height_range=(0.4, 0.7), # High
                        hole_width_range=(0.3, 0.4), # BUT VERY NARROW
                        fill=NEUTRAL_BG, p=1.0,
                        mask_fill_value=0,
                    ),
                    # 2. Horizontal occlusion (rock at bottom, surface at top)
                    A.CoarseDropout(
                        num_holes_range=(1, 2),
                        hole_height_range=(0.5, 0.7), # Narrow vertically
                        hole_width_range=(0.3, 0.5),    # Wide horizontally
                        fill=NEUTRAL_BG, p=1.0,
                        mask_fill_value=0,
                    ),
                    # 3. Imitation of bubbles or small debris
                    A.CoarseDropout(
                        num_holes_range=(3, 8),
                        hole_height_range=(0.02, 0.05),
                        hole_width_range=(0.02, 0.05),
                        fill=NEUTRAL_BG, p=1.0,
                        mask_fill_value=0,
                    ),
                ], p=0.4)

                train_pipeline.append(occlusion_op)

        # 4. Finalize
        train_pipeline.extend(base_norm_tensor)
        
        return A.Compose(train_pipeline)

    # --- Stats Output ---
    def _print_stats(self, stats: dict):
        print("\n" + "="*60)
        print("📊 DATASET FILTERING STATISTICS")
        print("="*60)
        print(f"Total RAW records (from cache/FO):  {stats.get('total_raw_records', 0)}")
        print(f"Skipped (label not in train_labels): {stats.get('skipped_by_label', 0)}")
        print("-" * 60)
        print(f"Annotations in TRAIN:                {stats.get('train_annotations', 0)}")
        print(f"Annotations in VAL:                  {stats.get('val_annotations', 0)}")
        print("-" * 60)
        print(f"Annotations in BOTH (leak!):         {stats.get('overlap_annotations', 0)}")
        print(f"Annotations SKIPPED (no train/val tag): {stats.get('skipped_by_tag', 0)}")
        print("="*60 + "\n")

    # --- Data Preparation ---

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Setting up dataset: {self.hparams.dataset_name}")

        cache_path = self._get_cache_path()
        all_records = []

        # 1. LOAD ALL DATA (either from cache or FiftyOne)
        if self.hparams.use_cache and os.path.exists(cache_path):
            logger.info(f"Loading ALL RAW records from cache: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
        else:
            logger.info("Extracting ALL records from FiftyOne...")
            fo_dataset = fo.load_dataset(self.hparams.dataset_name)
            all_records = self._extract_all_records(fo_dataset)
            
            if self.hparams.use_cache:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(all_records, f)
                logger.info(f"Raw records cache saved: {cache_path}")

        # 2. FILTERING AND SPLITTING IN MEMORY (Instant)
        logger.info("Filtering and splitting records by tags/labels...")
        train_records, val_records, stats = self._filter_and_split_records(all_records)
        del all_records

        self._print_stats(stats)
        logger.info(f"Classes: {self.num_classes}, train samples: {len(train_records)}")

        # 3. DATASET INITIALIZATION
        self.train_dataset = FishialDatasetOnlineCutting(
            records=train_records,
            img_size=self.hparams.image_size,
            train_state=True,
            transform=self.get_transform(is_train=True),
            instance_data=self.hparams.instance_data,
            bg_removal_prob=self.hparams.bg_removal_prob, 
            bbox_padding_limit=self.hparams.bbox_padding_limit,
            alignment_method=self.hparams.alignment_method
        )

        if val_records:
            self.val_dataset = FishialDatasetOnlineCutting(
                records=val_records,
                img_size=self.hparams.image_size,
                train_state=False,
                transform=self.get_transform(is_train=False),
                instance_data=self.hparams.instance_data,
                bg_removal_prob=self.hparams.bg_removal_prob, 
                bbox_padding_limit=self.hparams.bbox_padding_limit,
                alignment_method=self.hparams.alignment_method
            )

    # 💡 NEW METHOD: Extracts everything regardless of settings
    def _extract_all_records(self, ds) -> list:
        all_rec = []
        for sample in tqdm(ds, desc="Parsing FiftyOne (Global)"):
            image_id = sample.id if not sample.has_field('image_id') else sample['image_id']
            w, h = (sample.metadata.width, sample.metadata.height) if sample.metadata else (sample['width'], sample['height'])

            if sample.has_field("General body shape") and sample["General body shape"] is not None:
                for polyline in sample["General body shape"].polylines:
                    if not polyline.points:
                        continue

                    poly = [[int(p[0] * w), int(p[1] * h)] for p in polyline.points[0]]
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]

                    # Save "raw" record. ID mapping will be done later
                    all_rec.append({
                        'filepath': sample.filepath,
                        'poly': poly,
                        'bbox_xyxy': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                        'label': polyline.label,
                        'annotation_id': polyline['ann_id'] if polyline.has_field('ann_id') else None,
                        'drawn_fish_id': polyline['drawn_fish_id'] if polyline.has_field('drawn_fish_id') else None,
                        'image_id': image_id,
                        'tags': list(polyline.tags) if polyline.tags else [],
                    })
        return all_rec

    # 💡 NEW METHOD: Quickly filters cached data
    def _filter_and_split_records(self, all_records: list):
        train_rec, val_rec = [], []
        
        stats = {
            "total_raw_records": len(all_records),
            "skipped_by_label": 0,
            "train_annotations": 0,
            "val_annotations": 0,
            "overlap_annotations": 0,
            "skipped_by_tag": 0
        }

        # 1. Safely initialize sets
        # If None - leave as None. If list - convert to set.
        train_tags_set = set(self.hparams.train_tags) if self.hparams.train_tags is not None else None
        val_tags_set = set(self.hparams.val_tags) if self.hparams.val_tags is not None else None

        for rec in tqdm(all_records, desc="Filtering and splitting records"):
            label = rec['label']
            
            # 2. Label filter
            if label not in self.labels_to_train:
                stats["skipped_by_label"] += 1
                continue
                
            # Safely get ID from mapping
            mapping_info = self.class_mapping.get(label, {})
            
            # Create a copy of the record
            processed_rec = rec.copy()
            processed_rec['id_internal'] = mapping_info.get('id', -1)
            processed_rec['species_id'] = mapping_info.get('fishial_extra', {}).get('species_id')

            # 3. Safely extract image tags (in case 'tags' key is missing)
            tags = set(rec.get('tags', []))
            
            # 4. Distribution logic
            is_val = not tags.isdisjoint(val_tags_set) if val_tags_set is not None else True
            is_train = not tags.isdisjoint(train_tags_set) if train_tags_set is not None else True

            # 5. Handle overlaps and skips
            if is_val and is_train:
                stats["overlap_annotations"] += 1
                # If tags for both train and val are None, ALL images fall here.
                if train_tags_set is not None or val_tags_set is not None:
                    # logger.warning(f"Annotation {rec.get('annotation_id', 'Unknown')} (Label: {label}) has BOTH train and val tags.")
                    pass
            elif not is_val and not is_train:
                stats["skipped_by_tag"] += 1

            if is_val:
                val_rec.append(processed_rec)
                stats["val_annotations"] += 1
            if is_train:
                train_rec.append(processed_rec)
                stats["train_annotations"] += 1

        return train_rec, val_rec, stats

    # --- DataLoaders ---

    def train_dataloader(self) -> DataLoader:
        
        my_batch_sampler = BalancedBatchSampler(
            labels=self.train_dataset.targets,
            classes_per_batch=self.hparams.classes_per_batch,
            samples_per_class=self.hparams.samples_per_class,
            alpha=0.5,
        )
        
        train_kw = dict(
            batch_sampler=my_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self._instance_collate if self.hparams.instance_data else None,
        )
        if self.hparams.num_workers > 0:
            train_kw["prefetch_factor"] = int(self.hparams.prefetch_factor)
        return DataLoader(self.train_dataset, **train_kw)

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None
        nw = self.hparams.val_num_workers
        loader_kw = dict(
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self._instance_collate if self.hparams.instance_data else None,
        )
        if nw > 0:
            loader_kw["prefetch_factor"] = int(self.hparams.prefetch_factor)
        return DataLoader(self.val_dataset, **loader_kw)

    @staticmethod
    def _instance_collate(batch):
        """Custom collate that keeps metadata as a list instead of stacking."""
        images, labels, masks, metas = zip(*batch)
        return (
            default_collate(images),
            default_collate(labels),
            default_collate(masks),
            list(metas),
        )