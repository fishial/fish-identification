import torch
from jinja2 import Environment, FileSystemLoader
from utils import get_dataset_dicts, get_eval_on_selected_set

import cv2
import numpy as np
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2 import model_zoo

import os
from os import listdir
from os.path import isfile, join


def get_cfg_set(path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATASETS.TRAIN = ("fishial_Test",)
    cfg.DATASETS.TEST = ("fishial_Test")
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35
    return cfg


for d in ["Train", "Test"]:
    DatasetCatalog.register("fishial_" + d, lambda d=d: get_dataset_dicts("../fishial_collection", d))
    MetadataCatalog.get("fishial_" + d).set(thing_classes=["fish"], evaluator_type="coco")

fishial_metadata = MetadataCatalog.get("fishial_Train").set(thing_classes=["fish"], evaluator_type="coco")
path_to_old = "../best_scores/model_0059999_old_version.pth"
path_to_new = "../best_scores/model_0067499_amp_on.pth"

img_dir = 'test_images'
outpu_folder = "processed_images"
os.makedirs(outpu_folder, exist_ok=True)

cfg_old = get_cfg_set(path_to_old)
cfg_new = get_cfg_set(path_to_new)
predictor_old = DefaultPredictor(cfg_old)
predictor_new = DefaultPredictor(cfg_new)

df_test = get_eval_on_selected_set([(os.path.basename(path_to_old), cfg_old),
                                   (os.path.basename(path_to_new), cfg_new)],
                                  "fishial_Test")

imgs_test = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

list_of_paths = []
for img_path in imgs_test:
    im = cv2.imread(os.path.join(img_dir, img_path))
    outputs1 = predictor_old(im)
    v1 = Visualizer(im[:, :, ::-1], metadata=fishial_metadata, scale=0.8)
    v1 = v1.draw_instance_predictions(outputs1["instances"].to("cpu"))

    outputs2 = predictor_new(im)
    v2 = Visualizer(im[:, :, ::-1], metadata=fishial_metadata, scale=0.8)
    v2 = v2.draw_instance_predictions(outputs2["instances"].to("cpu"))

    vis = np.concatenate((v1.get_image()[:, :, ::-1], v2.get_image()[:, :, ::-1]), axis=1)
    new_path = os.path.join(outpu_folder, img_path)
    list_of_paths.append(new_path)
    cv2.imwrite(new_path, vis)

# 2. Create a template Environment
env = Environment(loader=FileSystemLoader('templates'))

# 3. Load the template from the Environment
template = env.get_template('report_template.html')

# 4. Render the template with variables
html = template.render(page_title_text='Fishial Report',
                       title_text='Fishial experiments report',
                       imgs_example=list_of_paths,
                       text='This is a report comparing two Mask RCNN models trained on detectron2 on different datasets, old and new fish collection.',
                       val_text='Testing the Model on Validation Data by AP metrics',
                       train_text='Testing the Model on Entire Data by AP metrics',
                       sp500_history=df_test,
                       sp500_history_summary=df_test)

# 5. Write the template to an HTML file
with open('html_report_jinja.html', 'w') as f:
    f.write(html)
