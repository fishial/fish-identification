import os
import sys

# # Import custom modules
# # Modify sys.path to include the root directory containing 'fish-identification'
# CURRENT_FOLDER_PATH = os.path.abspath(__file__)
# DELIMITER = 'fish-identification'
# pos = CURRENT_FOLDER_PATH.find(DELIMITER)
# if pos != -1:
#     sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
#     print("SETUP: sys.path updated")
    

from tqdm import tqdm

import logging
from typing import Optional

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
    """Encapsulates all data loading and processing logic."""
    def __init__(self, dataset_name: str, batch_size: int, classes_per_batch: int, samples_per_class: int, image_size: int, num_workers: int):
        super().__init__()
        # save_hyperparameters() saves arguments to self.hparams
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.label_to_id = {}
        self.num_classes = 0
        
    def _get_data_config(self, dataset):
        labels_dict = {}
        for sample in tqdm(dataset):
            base_name = os.path.basename(sample['filepath'])
            width = sample['width']
            height = sample['height']

            polyline = sample['polyline']
            
            if polyline['label'] not in labels_dict:
                labels_dict.update({polyline['label']: []})

            poly = [[int(point[0] * width), int(point[1] * height)] for point in polyline['points'][0]]
            labels_dict[polyline['label']].append({
                                'id':sample['annotation_id'],
                                'name': polyline['label'],
                                'base_name': base_name,
                                'image_id': sample['image_id'],
                                'poly': poly,
                                'file_name': sample['filepath']})
        return labels_dict

    def get_transform(self, is_train: bool):
        """Returns the appropriate data augmentation pipeline."""
        if is_train:
            return A.Compose([
                A.Resize(self.hparams.image_size, self.hparams.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.hparams.image_size, self.hparams.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def setup(self, stage: Optional[str] = None):
        """Called on every GPU to load datasets and create mappings."""
        logger.info(f"Setting up data for stage: {stage}, dataset: {self.hparams.dataset_name}")
        
        # In a real scenario, you would have a dataset in FiftyOne.
        # For demonstration, we'll create a dummy one if it doesn't exist.
        try:
            fo_dataset = fo.load_dataset(self.hparams.dataset_name)
        except ValueError:
            logger.warning(f"Dataset '{self.hparams.dataset_name}' not found. Creating a dummy dataset.")
            fo_dataset = fo.Dataset(self.hparams.dataset_name)
            # Add dummy data
            dummy_samples = []
            for i in range(20):
                filepath = f"/tmp/img_{i}.jpg"
                label = f"class_{i % 5}"
                tag = "train" if i < 15 else "val"
                sample = fo.Sample(filepath=filepath, ground_truth=fo.Classification(label=label), tags=[tag])
                dummy_samples.append(sample)
            fo_dataset.add_samples(dummy_samples)
            fo_dataset.persistent = True

        train_view = fo_dataset.match_tags("train")
        val_view = fo_dataset.match_tags("val")
        
        train_records = self._get_data_config(train_view)
        val_records = self._get_data_config(val_view)
        
        self.label_to_id = {label: idx for idx, label in enumerate(list(train_records))}
        self.num_classes = len(self.label_to_id)
        
        self.train_dataset = FishialDatasetFoOnlineCuting(
            train_records, self.label_to_id, transform=self.get_transform(is_train=True)
        )
        self.val_dataset = FishialDatasetFoOnlineCuting(
            val_records, self.label_to_id, transform=self.get_transform(is_train=False)
        )
        logger.info(f"Setup complete. Found {self.num_classes} classes.")
        logger.info(f"Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        """Creates the training dataloader with a balanced sampler."""
        
        # The BatchSampler will group the indices from MPerClassSampler into batches.
        # The batch_size here should be classes_per_batch * samples_per_class
        effective_batch_size = self.hparams.classes_per_batch * self.hparams.samples_per_class
        
        # The MPerClassSampler requires a batch size that is a multiple of `samples_per_class`.
        # However, it doesn't use the `batch_size` parameter from DataLoader directly.
        # Instead, it acts as a `batch_sampler`.
        m_per_class_sampler = samplers.MPerClassSampler(
            labels=self.train_dataset.targets,
            m=self.hparams.samples_per_class,
            batch_size=effective_batch_size,
            length_before_new_iter=len(self.train_dataset)
        )
        
        batch_sampler = BatchSampler(m_per_class_sampler, batch_size=effective_batch_size, drop_last=False)
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Creates the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
