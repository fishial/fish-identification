import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/fishial/Fishial/Object-Detection-Model')

import os
import cv2
import copy
import torch
import time
import logging
import datetime
import numpy as np
import albumentations as A

from module.segmentation_package.src.utils import run_eval_checkpoints
from module.segmentation_package.src.utils import get_current_date_in_format
from module.segmentation_package.src.utils import get_dataset_dicts
from module.segmentation_package.src.utils import get_prepared_data, get_empty_ann
from module.segmentation_package.src.utils import remove_tmp_files
from module.segmentation_package.src.utils import save_json
from module.segmentation_package.src.utils import split_ds
from module.segmentation_package.src.utils import get_fiftyone_dicts
from module.segmentation_package.src.trainer import Trainer

import albumentations as A

# copy paste source
from module.segmentation_package.src.copy_paste import CopyPaste
from module.segmentation_package.src.coco import CocoDetectionCP
from module.segmentation_package.src.CopyPasteCustom import apply_copy_paste_aug, get_copy_paste_instance, get_images_from_instance
from detectron2.data import build_detection_train_loader
from pycocotools import mask
from skimage import measure

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.projects.point_rend import ColorAugSSDTransform, add_pointrend_config
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultPredictor

import fiftyone as fo
import fiftyone.zoo as foz

setup_logger()

data_valid_ann = None

        
class MyMapper:
    """Mapper which uses `detectron2.data.transforms` augmentations"""

    def __init__(self, cfg, is_train: bool = True):
        self.cfg = cfg
        self.is_train = is_train

        mode = "training" if is_train else "inference"
        # print(f"[MyDatasetMapper] Augmentations used in {mode}: {self.augmentations}")

    def __call__(self, dataset_dict):
        torch.cuda.empty_cache()
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if len(dataset_dict['annotations']) > 0:
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
                    "iscrowd": 0}

                annos.append(obj)

            image_shape = image.shape[:2]  # h, w
            instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.cfg.INPUT.MASK_FORMAT)
            dataset_dict["instances"] = instances #utils.filter_empty_instances(instances)
        else:
            image = get_images_from_instance(dataset_dict)['image_full']
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            
            instances = utils.annotations_to_instances(dataset_dict['annotations'], 
                                                       (dataset_dict['width'], dataset_dict['height']),
                                                       mask_format=self.cfg.INPUT.MASK_FORMAT)
            dataset_dict["instances"] = instances
        return dataset_dict

class MyTrainer(DefaultTrainer):
    
#     @classmethod
#     def build_train_loader(cls, cfg, sampler=None):
#         return build_detection_train_loader(
#             cfg, mapper=MyMapper(cfg, True), sampler=sampler)
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        cfg.SOLVER.STEP += 1
        return COCOEvaluator(dataset_name, cfg, True, output_folder, cfg = cfg)
    
    
def main():
#     global data_valid_ann
#     path_to_imgs = r"dataset/fishial_collection/data"
#     path_to_coco_file = r"dataset/export/03_export_Verified_ALL.json"
#     path_to_empty_ann = r"/home/fishial/Fishial/dataset/coco_val2017"

#     data_full, _ = get_prepared_data(path_to_imgs, path_to_coco_file)
#     data_empty = get_empty_ann(path_to_empty_ann)
#     train_folder = "Train"
#     for d in ["Train", "Test"]:
#         DatasetCatalog.register("fishial_" + d, lambda d=d: split_ds(data_full,data_empty, d))
#         MetadataCatalog.get("fishial_" + d).set(thing_classes=["fish"], evaluator_type="coco")

    dataset = fo.load_dataset('fishial-dataset-november-2022')
    for d in ["train", "val"]:
        view = dataset.match_tags(d)
        DatasetCatalog.register("fishial_" + d, lambda view=view: get_fiftyone_dicts(view))
        MetadataCatalog.get("fishial_" + d).set(thing_classes=["fish"])
    
#     dataset_train = split_ds(data_full,data_empty, "Train")
#     data_valid_ann = get_copy_paste_instance(dataset_train)

    config_path = "/home/fishial/Fishial/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    experiment_name = "citiscape"
    main_folder = "output"

    learning_rates = [0.0028]  # .01 .001 .0001 .00025 .000025
    roi_batch_sizes = [128]  # 32, 64, 128, 256
    freezing_layers = [2]  # 0 1 2 3 4 5
    amp = [True]
    crop_enables = [True]
    max_iters = 2000000
    check_point_step = 50000
    
    DEVIDE = 50

    # Tune hyperparams with val set.
    for freezing_layer in freezing_layers:
        for learning_rate in learning_rates:
            for crop_enable in crop_enables:
                for amp_state in amp:
                    for roi_batch_size in roi_batch_sizes:
                        
                        torch.cuda.empty_cache()
                        experiment_folder = os.path.join(main_folder,
                                                         os.path.join(experiment_name, get_current_date_in_format()))
                        # Create experiment folder
                        os.makedirs(experiment_folder, exist_ok=True)

                        cfg = get_cfg()
                        add_pointrend_config(cfg)
                        
                        cfg.merge_from_file(config_path)
                        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
                        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
                        
                        cfg.DATASETS.TRAIN = ("fishial_train", )
                        cfg.DATASETS.TEST = ("fishial_val", )
                        
                        cfg.DATALOADER.NUM_WORKERS = 4

                        cfg.MODEL.WEIGHTS = "/home/fishial/Fishial/saved_models/model_final_115bfb.pkl"
                        
                        cfg.INPUT.CROP.ENABLED = True
                        cfg.TEST.AUG.ENABLED = True
                        cfg.TEST.EVAL_PERIOD = check_point_step
                        
                        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
                        cfg.OUTPUT_DIR = experiment_folder
                        
                        cfg.SOLVER.STEP = 0 # init step always 0
                        cfg.SOLVER.IMS_PER_BATCH = 6
                        cfg.SOLVER.MAX_ITER = max_iters
                        cfg.SOLVER.CHECKPOINT_PERIOD = check_point_step
                        cfg.SOLVER.AMP.ENABLED = amp_state
                        cfg.SOLVER.IMS_PER_BATCH = 6
                        cfg.SOLVER.BASE_LR = learning_rate
                        cfg.SOLVER.GAMMA = 0.9
                        cfg.SOLVER.STEPS = [int(i * (max_iters/DEVIDE)) for i in range(1, DEVIDE)]
                        cfg.SOLVER.WARMUP_ITERS = 0
                        cfg.SOLVER.WARMUP_FACTOR = 0
                        
                        # cfg.MODEL.FPN.OUT_CHANNELS = 128
                        cfg.MODEL.BACKBONE.FREEZE_AT = 1
                        # cfg.MODEL.POINT_HEAD.NUM_FC = 3
                        cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 6
                        
#                         cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 128
#                         cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 512
#                         cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 512

#                         cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch_size
                        trainer = MyTrainer(cfg)

                        trainer.resume_or_load()
                        trainer.train()
                        
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
