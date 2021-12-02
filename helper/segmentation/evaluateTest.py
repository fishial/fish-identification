import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import os

from module.segmentation_package.src.utils import save_to_json, save_json, read_json
from module.segmentation_package.src.utils import beautifier_results
from module.segmentation_package.src.utils import get_dataset_dicts
from os.path import isfile, join
from os import listdir

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg

from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer

for d in ["Train", "Test"]:
    DatasetCatalog.register("fishial_" + d, lambda d=d: get_dataset_dicts("../fishial_collection", d))
    MetadataCatalog.get("fishial_" + d).set(thing_classes=["fish"], evaluator_type="coco")

input_folder = 'output_aug_3'
output_folder = "output_eval"

json_path = os.path.join(input_folder, "score_of_models.json")

os.makedirs(output_folder, exist_ok=True)

list_of_files_in_directory = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
array_of_eval_results = []

for file_name in list_of_files_in_directory:
    splited = os.path.splitext(file_name)
    for data_set in ['fishial_Test']:
        
        if splited[1] != '.pth':
            continue
            
        mem_eval = read_json(json_path)
        if mem_eval is None:
            mem_eval = {}

        checker = False

        if file_name not in mem_eval:
            checker = True
        elif data_set not in mem_eval[file_name]:
            checker = True

        if checker:
            print("Update")
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.DATASETS.TRAIN = (data_set,)
            cfg.DATASETS.TEST = (data_set)
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.MODEL.WEIGHTS = os.path.join(input_folder, file_name)
            cfg.SOLVER.IMS_PER_BATCH = 2
            cfg.MODEL.DEVICE = "cuda"
            cfg.OUTPUT_DIR = output_folder
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            trainer = DefaultTrainer(cfg)
            trainer.resume_or_load(resume=True)
            evaluator = COCOEvaluator(data_set, cfg, False, output_dir=output_folder)
            val_loader = build_detection_test_loader(cfg, data_set)
            value_sd = inference_on_dataset(trainer.model, val_loader, evaluator)

            mem_eval.update({
                file_name: {
                    data_set: value_sd
                }
            })
            save_json(mem_eval, json_path)
            array_of_eval_results.append([file_name, value_sd])
            print(beautifier_results(array_of_eval_results))
print(beautifier_results(array_of_eval_results))

