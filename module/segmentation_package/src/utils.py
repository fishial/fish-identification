import os
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from os.path import isfile, join
from os import listdir

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer


def read_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        return data
    else:
        return None


def get_current_date_in_format():
    now = datetime.now()
    return now.strftime("%d_%m_%Y_%H_%M_%S")


def get_dataset_dicts(img_dir, state, json_file="fishial_collection_correct.json"):
    json_file = os.path.join(img_dir, json_file)
    with open(json_file) as f:
        data = json.load(f)

    bodyes_shapes_ids = []
    for i in data['categories']:
        if i['name'] == 'General body shape':
            bodyes_shapes_ids.append(int(i['id']))

    dataset_dicts = []

    img_dir = os.path.join(img_dir, state)
    for i in tqdm(range(len(data['images']))):
        if 'train_data' in data['images'][i]:
            
            state_json = 'Train' if data['images'][i]['train_data'] else 'Test'

            if state != state_json:
                continue
            record = {}
            filename = os.path.join(img_dir, data['images'][i]['file_name'])
            width, height = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["height"] = width
            record["width"] = height
            record["image_id"] = i

            objs = []

            for ann in data['annotations']:
                if 'segmentation' in ann and ann['image_id'] == data['images'][i]['id'] and ann[
                    'category_id'] in bodyes_shapes_ids:
                    px = []
                    py = []
                    for z in range(int(len(ann['segmentation'][0]) / 2)):
                        px.append(ann['segmentation'][0][z * 2])
                        py.append(ann['segmentation'][0][z * 2 + 1])

                    obj = {
                        "bbox": [np.min(px).tolist(), np.min(py).tolist(), np.max(px).tolist(), np.max(py).tolist()],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": ann['segmentation'],
                        "category_id": 0,
                        "iscrowd": 0
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def get_dataset_dicts_sep(img_dir, json_file):
    
    with open(json_file) as f:
        data = json.load(f)

    bodyes_shapes_ids = []
    for i in data['categories']:
        if i['name'] == 'General body shape':
            bodyes_shapes_ids.append(int(i['id']))

    dataset_dicts = []

    for i in tqdm(range(len(data['images']))):
        if 'train_data' in data['images'][i]:
     
            record = {}
            filename = os.path.join(img_dir, data['images'][i]['file_name'])
            width, height = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["height"] = width
            record["width"] = height
            record["image_id"] = data['images'][i]['id']

            objs = []

            for ann in data['annotations']:
                if 'segmentation' in ann and ann['image_id'] == data['images'][i]['id'] and ann[
                    'category_id'] in bodyes_shapes_ids:
                    px = []
                    py = []
                    for z in range(int(len(ann['segmentation'][0]) / 2)):
                        px.append(ann['segmentation'][0][z * 2])
                        py.append(ann['segmentation'][0][z * 2 + 1])

                    obj = {
                        "bbox": [np.min(px).tolist(), np.min(py).tolist(), np.max(px).tolist(), np.max(py).tolist()],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": ann['segmentation'],
                        "category_id": 0,
                        "iscrowd": 0
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def beautifier_results(results):
    line = 70 * "="
    header = line + """\n|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  Name 
    \n|:------:|:------:|:------:|:------:|:------:|:------:|:------: """
    for result in results:
        header += """\n| {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |{}""".format(
            result[1]['segm']['AP'],
            result[1]['segm']['AP50'],
            result[1]['segm']['AP75'],
            result[1]['segm']['APs'],
            result[1]['segm']['APm'],
            result[1]['segm']['APl'],
            result[0])
    return header + "\n" + line


def save_to_json(results, mypath):
    json_file = {}
    for result in results:
        single_rec = {
            result[0]: {
                'AP': result[1]['segm']['AP'],
                'AP50': result[1]['segm']['AP50'],
                'AP75': result[1]['segm']['AP75'],
                'APs': result[1]['segm']['APs'],
                'APm': result[1]['segm']['APm'],
                'APl': result[1]['segm']['APl']}}
        json_file.update(single_rec)
    total_path = os.path.join(mypath, "eval_scores.json")
    save_json(json_file, total_path)
    return json_file


def save_json(object, path):
    with open(path, 'w') as f:
        json.dump(object, f)


def custom_config(num_classes=1, train_name="fishial_Train", test_name="fishial_Test", output_dir="output"):
    output_dir = os.path.join(output_dir, get_current_date_in_format())

    cfg = get_cfg()
    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Model
    #     cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    #     cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    #     cfg.MODEL.RESNETS.DEPTH = 34

    # Solver
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 35000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500

    # Test
    #     cfg.TEST.DETECTIONS_PER_IMAGE = 40
    cfg.TEST.EVAL_PERIOD = 35000

    # # INPUT
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)

    # DATASETS
    cfg.DATASETS.TEST = (test_name,)
    cfg.DATASETS.TRAIN = (train_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.OUTPUT_DIR = output_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def get_eval_on_selected_set(cfgs, set):
    output_folder = "./output_eval/"
    os.makedirs(output_folder, exist_ok=True)
    result = {'model_name': []}
    for i in cfgs:
        trainer = DefaultTrainer(i[1])
        evaluator = COCOEvaluator(set, i[1], False, output_dir="./{}/".format(output_folder))
        val_loader = build_detection_test_loader(i[1], set)
        value_sd = inference_on_dataset(trainer.model, val_loader, evaluator)

        for key in value_sd['segm']:
            if key in result:
                result[key].append(value_sd['segm'][key])
            else:
                result.update({key: [value_sd['segm'][key]]})
        result['model_name'].append(i[0])
    df = pd.DataFrame.from_dict(result)
    df['SUM'] = df['AP'] + df['AP50'] + df['AP75'] + \
                df['APs'] + df['APm'] + df['APl']
    sum_c = df['SUM']
    df.drop(labels=['SUM'], axis=1, inplace=True)
    df.insert(0, 'SUM', sum_c)
    return df


def run_eval_checkpoints(cfg, input_folder, test_dataset):
    tmp_folder = os.path.join(os.path.join(input_folder, ".."), "tmp_folder")

    os.makedirs(tmp_folder, exist_ok=True)

    list_of_files_in_directory = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
    array_of_eval_results = []

    for file_name in list_of_files_in_directory:
        splited = os.path.splitext(file_name)
        if splited[1] == '.pth':
            cfg.MODEL.WEIGHTS = os.path.join(input_folder, file_name)
            trainer = DefaultTrainer(cfg)
            trainer.resume_or_load(resume=True)
            evaluator = COCOEvaluator(test_dataset, cfg, False, output_dir=tmp_folder)
            val_loader = build_detection_test_loader(cfg, "fishial_Test")
            value_sd = inference_on_dataset(trainer.model, val_loader, evaluator)
            array_of_eval_results.append([file_name, value_sd])
    save_to_json(array_of_eval_results, input_folder)


def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("[CLEANER] File not found in the directory")


def remove_tmp_files(path):
    files_to_remove = ["last_checkpoint", 'metrics.json', "model_final.pth"]
    list_of_files_in_directory = [f for f in listdir(path) if isfile(join(path, f))]

    for file_name in list_of_files_in_directory:
        splited = file_name.split(".")
        if splited[0] == 'events':
            files_to_remove.append(file_name)

    for i in files_to_remove:
        remove_file(os.path.join(path, i))


def get_mask(image, pts):
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = image[y:y + h, x:x + w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst


def approximate(output, eps):
    polygons = []
    for z in range(len(output['instances'])):
        masks = np.array(output['instances'].get('pred_masks')[z].to('cpu'))
        imgUMat = np.array(masks * 255, dtype=np.uint8)
        cnts, hierarchy = cv2.findContours(imgUMat, 1, 2)
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
        cnts_s = cnts[len(cnts) - 1]
        epsilon = eps * cv2.arcLength(cnts_s, True)
        approx = cv2.approxPolyDP(cnts_s, epsilon, True)
        polygons_dict = []
        for i in range(len(approx)):
            polygons_dict.append([int(approx[i][0][0]), int(approx[i][0][1])])
        polygons.append(polygons_dict)

    return polygons