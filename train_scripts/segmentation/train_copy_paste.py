import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import os
import cv2
import copy
import torch
import numpy as np
import albumentations as A

# copy paste source
from module.segmentation_package.src.copy_paste import CopyPaste
from module.segmentation_package.src.coco import CocoDetectionCP
from module.segmentation_package.src.CopyPasteCustom import apply_copy_paste_aug, get_copy_paste_instance
from module.segmentation_package.src.utils import get_dataset_dicts_sep, get_dataset_dicts

from pycocotools import mask
from skimage import measure

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode

setup_logger()

DatasetCatalog.clear()

for d in ["Train", "Test"]:
        DatasetCatalog.register("fishial_" + d, lambda d=d: get_dataset_dicts('FishialReaserch/datasets/fishial_collection/cache', d, json_file="FishialReaserch/datasets/fishial_collection/export.json"))
        MetadataCatalog.get("fishial_" + d).set(thing_classes=["fish"], evaluator_type="coco")
dataset_dicts_train = DatasetCatalog.get("fishial_Train")

dataset_train = get_dataset_dicts('FishialReaserch/datasets/fishial_collection/cache', 'Train', json_file="FishialReaserch/datasets/fishial_collection/export.json")
data_valid_ann = get_copy_paste_instance(dataset_train)


class MyMapper:
    """Mapper which uses `detectron2.data.transforms` augmentations"""

    def __init__(self, cfg, is_train: bool = True):

        self.is_train = is_train

        mode = "training" if is_train else "inference"
        # print(f"[MyDatasetMapper] Augmentations used in {mode}: {self.augmentations}")

    def __call__(self, dataset_dict):
        torch.cuda.empty_cache()
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        aug_sample = apply_copy_paste_aug(dataset_dict, data_valid_ann)

        image = aug_sample['image']
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        polygons = aug_sample['segmentation']

        annos = []

        for polygon in polygons:
            coco_poly = []
            px = []
            py = []
            for pts in polygon:
                px.append(pts[0])
                py.append(pts[1])
                coco_poly.append(pts[0])
                coco_poly.append(pts[1])

            obj = {
                "bbox": [np.min(px).tolist(), np.min(py).tolist(), np.max(px).tolist(), np.max(py).tolist()],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [coco_poly],
                "category_id": 0,
                "iscrowd": 0
            }

            annos.append(obj)

        image_shape = image.shape[:2]  # h, w
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper=MyMapper(cfg, True), sampler=sampler
        )


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("fishial_Train",)
# cfg.INPUT.FORMAT = 'BGR'
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = 'output_normal_custom_schedule_lr/model_0272499.pth'
# "../best_scores/model_0067499_amp_on.pth" # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 2  # increase it
cfg.SOLVER.BASE_LR = 0.0009
cfg.SOLVER.GAMMA = 0.9
DEVIDE = 30
STEPS = 500000
cfg.SOLVER.STEPS = [int(i * (STEPS/DEVIDE)) for i in range(1, DEVIDE)]
# The iteration number to decrease learning rate by GAMMA.

cfg.SOLVER.WARMUP_ITERS = 0
cfg.SOLVER.WARMUP_FACTOR = 0

cfg.SOLVER.AMP.ENABLED = True
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.SOLVER.CHECKPOINT_PERIOD = 2500
cfg.SOLVER.MAX_ITER = STEPS
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.OUTPUT_DIR = 'output_aug_custom_schedule_lr'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.INPUT.CROP.ENABLED = True
cfg.TEST.AUG.ENABLED = True

trainer = MyTrainer(cfg)
trainer.resume_or_load()
trainer.train()