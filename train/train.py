import torch, torchvision
print('Torch version: {}'.format(torch.__version__))
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
# create Datest
import os
import json
from detectron2.structures import BoxMode
import itertools
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import convert_to_coco_json
import cv2


# write a function that loads the dataset into detectron2's standard format
def get_dataset_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for _, v in imgs_anns.items():
        if len(v["regions"]['0']['shape_attributes']['all_points_x']) < 20:
            continue
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        width, height = cv2.imread(filename).shape[:2]
        record["file_name"] = v["filename"]
        record["height"] = width
        record["width"] = height

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]

            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("fishial/" + d, lambda d=d: get_dataset_dicts("fishial/" + d))
    MetadataCatalog.get("fishial/" + d).set(thing_classes=["fishial"])
    convert_to_coco_json("fishial/" + d, 'fishial/' + d + '.json')

register_coco_instances("fishial/train-coco", {}, "fishial/train.json",
                        "fishial/train")

register_coco_instances("fishial/val-coco", {}, "fishial/val.json",
                        "fishial/val")

balloon_metadata_train = MetadataCatalog.get("fishial/train-coco")
balloon_metadata_val = MetadataCatalog.get("fishial/val-coco")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer

class MyTrainer(DefaultTrainer):

  # Uncomment if you want to made evalute VAL sets on COCOEvalutor
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = r"drive/My Drive/dataset/output/new29.230-42.681/model_final.pth"

cfg.DATASETS.TRAIN = ("fishial/train-coco",)
cfg.DATASETS.TEST = ("fishial/val-coco",)
cfg.TEST.EVAL_PERIOD = 2000
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00028
cfg.SOLVER.MAX_ITER = 23000
# cfg.MODEL.FPN.FUSE_TYPE = "sum" #'sum' or 'avg'
# cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

cfg.SOLVER.STEPS = (10000,)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()