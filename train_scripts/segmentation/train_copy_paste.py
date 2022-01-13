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

dataset_train = get_dataset_dicts('data/cache', 'Train', json_file="data/export.json")
data_valid_ann = get_copy_paste_instance(dataset_train)

DatasetCatalog.register("fishial_Train", dataset_train)
MetadataCatalog.get("fishial_Train").set(thing_classes=["fish"], evaluator_type="coco")

dataset_test = get_dataset_dicts('data/cache', 'Test', json_file="data/export.json")
DatasetCatalog.register("fishial_Test", dataset_train)
MetadataCatalog.get("fishial_Test").set(thing_classes=["fish"], evaluator_type="coco")

dataset_dicts_train = DatasetCatalog.get("fishial_Train")
train_metadata = MetadataCatalog.get("fishial_Train")


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
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        bboxes = aug_sample['bboxes']
        box_classes = np.array([b[-2] for b in bboxes])
        boxes = np.stack([b[:4] for b in bboxes], axis=0)
        mask_indices = np.array([b[-1] for b in bboxes])

        masks = aug_sample['masks']

        annos = []

        for enum, index in enumerate(mask_indices):
            curr_mask = masks[index]

            fortran_ground_truth_binary_mask = np.asfortranarray(curr_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(curr_mask, 0.5)

            annotation = {
                "segmentation": [],
                "iscrowd": 0,
                "bbox": ground_truth_bounding_box.tolist(),
                "category_id": 0,
                "bbox_mode": BoxMode.XYWH_ABS
            }
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                annotation["segmentation"].append(segmentation)

            annos.append(annotation)

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
cfg.DATASETS.VAL = ("fishial_Test",)
# cfg.INPUT.FORMAT = 'BGR'
cfg.DATASETS.TEST = ("fishial_Test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "model_0067499_amp_on-Copy1.pth"
# "../best_scores/model_0067499_amp_on.pth" # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 2  # increase it
cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.GAMMA = 0.1
# cfg.SOLVER.STEPS = (4000,)
# The iteration number to decrease learning rate by GAMMA.

# cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
# cfg.SOLVER.WARMUP_ITERS = 500
# cfg.SOLVER.WARMUP_METHOD = "linear"

cfg.SOLVER.AMP.ENABLED = True
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.SOLVER.CHECKPOINT_PERIOD = 2500
cfg.SOLVER.MAX_ITER = 100000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.OUTPUT_DIR = 'output_aug_3'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(cfg.dump())

trainer = MyTrainer(cfg)
trainer.resume_or_load()
trainer.train()
