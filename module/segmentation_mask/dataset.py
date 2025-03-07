import os
import random
import torch
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class RemoveFirstTwoChannels:
    """Custom transform to remove the first two channels from an image."""

    def __call__(self, pil_img):
        image_np = np.array(pil_img)

        if image_np.shape[-1] == 3:  # Ensure it's a 3-channel image
            image_np_new = np.expand_dims(image_np[:, :, 0], 0)
            return image_np_new

        return pil_img


class FishialFishDataset(torch.utils.data.Dataset):
    """Dataset for loading fish segmentation images and masks."""

    def __init__(self, imgs_dir, masks_dir, image_size=256, aug=False):
        self.image_size = image_size
        self.images_fps = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]
        self.masks_fps = [os.path.join(masks_dir, img) for img in os.listdir(masks_dir)]

        # Define transformations for images and masks
        self.transform_img = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.BILINEAR),
            RemoveFirstTwoChannels()
        ])

        # Augmentation pipeline
        self.augmentation = albu.Compose([
            albu.Blur(blur_limit=3, p=1),
            albu.RandomBrightnessContrast(p=0.2),
            albu.RandomRotate90(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
        ]) if aug else None

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        image = Image.open(self.images_fps[idx]).convert("RGB")
        mask = Image.open(self.masks_fps[idx])

        sample = {"image": image, "mask": mask}

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(**sample)

        return sample


class SimpleFishialFishDataset(FishialFishDataset):
    """Extended dataset that applies additional transformations to images and masks."""

    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)
        sample["image"] = self.transform_img(sample["image"])
        sample["mask"] = self.transform_mask(sample["mask"])
        return sample


class FishialSegmentDatasetFoOnlineCutting(torch.utils.data.Dataset):
    """Dataset for fish segmentation with FiftyOne dataset and polygon-based mask cutting."""

    def __init__(self, data_fo, image_size, aug=False, train_state=False, limit=0.1, min_poly_size=(50, 50)):
        self.data_fo = data_fo
        self.image_size = image_size
        self.train_state = train_state
        self.limit = limit
        self.min_poly_width, self.min_poly_height = min_poly_size

        self.transform_img = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Albumentations-based augmentation
        self.augmentation = albu.Compose([
            albu.Blur(blur_limit=3, p=1),
            albu.RandomBrightnessContrast(p=0.2),
            albu.RandomRotate90(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
        ]) if aug else None

        # Prepare dataset
        self.data = self._prepare_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        img_path, box_xywh, polygon, width_img, height_img = (
            sample_data["file_path"], sample_data["poly_box"], sample_data["polygon"],
            sample_data["width_img"], sample_data["height_img"]
        )

        image_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.fromarray(self._generate_mask(polygon, height_img, width_img))

        # Adjust bounding box and crop images
        box_xyxy = self._adjust_bbox(box_xywh, width_img, height_img)
        image_pil = image_pil.crop(box_xyxy)

        # Resize mask
        mask_resized = mask_pil.crop(box_xyxy).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask_resized = np.expand_dims(np.array(mask_resized), 0)

        sample = {"image": image_pil, "mask": mask_resized}

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(**sample)

        try:
            sample["image"] = self.transform_img(sample["image"])
        except Exception as e:
            print(f"Error transforming image: {e}")

        return sample

    def _prepare_data(self):
        """Prepare dataset by filtering and extracting bounding box data."""
        processed_data = []
        skip_count = 0
        
        for sample in tqdm(self.data_fo, desc="Processing dataset"):
            width, height = sample["width"], sample["height"]
            polylines = sample["General body shape"].polylines

            for polyline in polylines:
                polygon = np.array([[int(p[0] * width), int(p[1] * height)] for p in polyline.points[0]])

                # Compute bounding box
                min_x, min_y = np.min(polygon[:, 0]), np.min(polygon[:, 1])
                max_x, max_y = np.max(polygon[:, 0]), np.max(polygon[:, 1])
                poly_box = (min_x, min_y, max_x - min_x, max_y - min_y)

                # Skip small bounding boxes
                if poly_box[2] < self.min_poly_width or poly_box[3] < self.min_poly_height:
                    skip_count += 1
                    continue

                processed_data.append({
                    "file_path": sample.filepath,
                    "polygon": polygon,
                    "poly_box": poly_box,
                    "width_img": width,
                    "height_img": height
                })

        print(f"Total skipped small boxes: {skip_count}")
        return processed_data

    def _generate_mask(self, polygon, height, width):
        """Generate mask from polygon coordinates."""
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.reshape((-1, 1, 2))], 1)
        return mask

    def _adjust_bbox(self, bbox, img_width, img_height):
        """Adjust bounding box with random offset."""
        x, y, w, h = bbox
        x_shift = int(random.uniform(-self.limit * w, self.limit * w))
        y_shift = int(random.uniform(-self.limit * h, self.limit * h))

        new_x, new_y = max(0, x + x_shift), max(0, y + y_shift)
        new_w, new_h = min(w, img_width - new_x), min(h, img_height - new_y)

        return [new_x, new_y, new_x + new_w, new_y + new_h]