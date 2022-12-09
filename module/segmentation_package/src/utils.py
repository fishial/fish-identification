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

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.core.utils as fou

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


def get_image_class(data, image_id):
    for i in data['images']:
        if i['id'] == image_id:
            return i


def get_labels(json_file):
    data = read_json(json_file)

    bodyes_shapes_ids = []
    for i in data['categories']:
        if i['name'] == 'General body shape':
            bodyes_shapes_ids.append(i['supercategory'])
    return bodyes_shapes_ids


def get_anns_by_image_id(data, image_id):
    anns = []
    for ann_id in range(len(data['annotations']) - 1, -1, -1):
        if data['annotations'][ann_id]['image_id'] == image_id:
            anns.append(data['annotations'][ann_id])
            del data['annotations'][ann_id]
    return anns


def get_sorted_data(data):
    image_ids_dict = {}
    for ann_id in range(len(data['annotations']) - 1, -1, -1):
        if data['annotations'][ann_id]['image_id'] in image_ids_dict:
            image_ids_dict[data['annotations'][ann_id]['image_id']].append(data['annotations'][ann_id])
        else:
            image_ids_dict.update({
                data['annotations'][ann_id]['image_id']: [data['annotations'][ann_id]]})
    return image_ids_dict

def split_ds(ds,ds_empty, state):
    
    train_conf, test_conf = 0.75, 0.25
    
    if state == 'Train':
        start_idx_main, end_idx_main = 0, int(len(ds) * train_conf)
        start_idx_empty, end_idx_empty = 0, int(len(ds_empty) * train_conf)
        non_empty = [rec[1] for rec in list(ds.items())[start_idx_main:end_idx_main]] 
        empty = [rec[1] for rec in list(ds_empty.items())[start_idx_empty:end_idx_empty]]
#         non_empty.extend(empty)
        return non_empty
    else:
        start_idx_main, end_idx_main = int(len(ds) * train_conf + 1), len(ds)
        start_idx_empty, end_idx_empty = int(len(ds_empty) * train_conf + 1), len(ds_empty)
        non_empty = [rec[1] for rec in list(ds.items())[start_idx_main:end_idx_main]] 
        empty = [rec[1] for rec in list(ds_empty.items())[start_idx_empty:end_idx_empty]]
#         non_empty.extend(empty)
        return non_empty
        
#     non_empty = [rec[1] for rec in list(ds.items())[start_idx_main:end_idx_main]] 
#     empty = [rec[1] for rec in list(ds_empty.items())[start_idx_empty:end_idx_empty]]
#     non_empty.extend(empty)
#     return non_empty


def get_empty_ann(path_to_empty_ann):
    tmp_data = {}
    for f in listdir(path_to_empty_ann):

        if not isfile(join(path_to_empty_ann, f)): continue

        filename = os.path.join(path_to_empty_ann, f)
        try:
            width, height = cv2.imread(filename).shape[:2]
        except:
            print("skip file read error: ", filename)
            continue

        tmp_data.update({f: {
            "file_name": filename,
            "height": width,
            "width": height,
            "image_id": f,
            "annotations": []
        }})
    return tmp_data
    
def get_prepared_data(img_dir, json_file, path_to_class = None):
    data = read_json(json_file)
    if path_to_class:
        valid_labels = read_json(path_to_class)
        local_id_dict = {valid_labels[label_id]: int(label_id) for label_id in valid_labels}
    else:
        local_id_dict = {"Fish": 0}
    
    bodyes_shapes_ids = {}
    for i in data['categories']:
        if i['name'] == 'General body shape':
            bodyes_shapes_ids.update({int(i['id']): i['supercategory']})

    skip_data = []
    tmp_data = {}
    data_with_full_ann = get_sorted_data(data)

    for indices, image_dict in enumerate(data['images']):
        if image_dict['fishial_extra']['test_image'] or image_dict['fishial_extra']['xray'] or \
                image_dict['fishial_extra'][
                    'not_a_real_fish']:
            continue

        print(f"Left: {len(data['images']) - indices} skip: {len(skip_data)}", end='\r')
        #         if indices > 1000: continue
        if image_dict['id'] not in data_with_full_ann: continue
        anns = data_with_full_ann[image_dict['id']]
        if len(anns) == 0: continue

        filename = os.path.join(img_dir, image_dict['file_name'])
#         try:
#             width, height = cv2.imread(filename).shape[:2]
#             data['images'][indices]['width'] = width
#             data['images'][indices]['height'] = height
#         except:
#             print("skip file read error: ", filename)
#             continue

        tmp_data.update({image_dict['id']: {

            "file_name": filename,
            "height": data['images'][indices]['width'],
            "width": data['images'][indices]['height'],
            "image_id": image_dict['id'],
            "annotations": []
        }})

        for ann in anns:
            if 'category_id' not in ann: continue
            if 'segmentation' not in ann: continue
            if ann['category_id'] not in bodyes_shapes_ids: continue

            # some if conditional if we need manualy skip annotations
            px = []
            py = []
            for z in range(int(len(ann['segmentation'][0]) / 2)):
                px.append(ann['segmentation'][0][z * 2])
                py.append(ann['segmentation'][0][z * 2 + 1])

            bbox = [np.min(px).tolist(), np.min(py).tolist(), np.max(px).tolist(), np.max(py).tolist()]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann['segmentation'],
                "category_id": 0,
                "category_label": 'Fish',
                "annotation_id": ann['id'],
                "iscrowd": 0
            }
            tmp_data[ann['image_id']]['annotations'].append(obj)
    #     save_json(data, json_file)
    return tmp_data, [label for label in local_id_dict]


def get_dataset_dicts(tmp_data, state):
    list_of_ids = list(set([ann['category_id'] for asd in tmp_data for ann in tmp_data[asd]['annotations']]))
    list_of_img_in_class = {category_id: [] for category_id in list_of_ids}

    for img in tmp_data:
        for ann in tmp_data[img]['annotations']:
            list_of_img_in_class[ann['category_id']].append(img)

    min_eval_img = 10
    max_eval_percent = 0.2
    list_of_test = []
    list_of_train = []

    for i in list_of_img_in_class:
        eval_img = max(min_eval_img, int(len(list_of_img_in_class[i]) * max_eval_percent))
        list_of_test.extend(list_of_img_in_class[i][:eval_img])
        list_of_train.extend(list_of_img_in_class[i][eval_img:])

    list_of_train = list(set(list_of_train))

    list_of_test = list(set(list_of_test))
    list_of_test = [img_id for img_id in list_of_test if img_id not in list_of_train]

    dataset_dicts = []
    if state == 'Train':
        for image_id in list_of_train:
            dataset_dicts.append(tmp_data[image_id])
    else:
        for image_id in list_of_test:
            dataset_dicts.append(tmp_data[image_id])
    return dataset_dicts


def get_fiftyone_dataset(img_dir, json_file):
    samples = []

    data = read_json(json_file)
    local_id_dict = {"Fish": 0}
    
    bodyes_shapes_ids = {}
    for i in data['categories']:
        if i['name'] == 'General body shape':
            bodyes_shapes_ids.update({int(i['id']): i['supercategory']})

    skip_data = []
    data_with_full_ann = get_sorted_data(data)

    for indices, image_dict in enumerate(data['images']):
        
        if image_dict['fishial_extra']['test_image'] or image_dict['fishial_extra']['xray'] or \
                image_dict['fishial_extra'][
                    'not_a_real_fish']:
            skip_data.append(1)
            continue
        print(f"Left: {len(data['images']) - indices} skip: {len(skip_data)}", end='\r')
        
        if image_dict['id'] not in data_with_full_ann: continue
        anns = data_with_full_ann[image_dict['id']]
        if len(anns) == 0: continue

        filename = os.path.join(img_dir, image_dict['file_name'])
        
        width = data['images'][indices]['width']
        height = data['images'][indices]['height']
        odm_tag = 'include_in_odm' if data['images'][indices]['fishial_extra']['include_in_odm'] else 'not_include_in_odm'
        
        
        polylines = []
        detections = []
        res = False
        for ann in anns:
            
            if 'category_id' not in ann: continue
            if 'segmentation' not in ann: continue
            if ann['category_id'] not in bodyes_shapes_ids: continue

            # some if conditional if we need manualy skip annotations
            px = []
            py = []
            poly = []
            for z in range(int(len(ann['segmentation'][0]) / 2)):
                px.append(ann['segmentation'][0][z * 2]/width)
                py.append(ann['segmentation'][0][z * 2 + 1]/height)
                
                poly.append((ann['segmentation'][0][z * 2]/width, ann['segmentation'][0][z * 2 + 1]/height))

            bbox = [np.min(px).tolist(), np.min(py).tolist(), np.max(px).tolist(), np.max(py).tolist()]
            label = bodyes_shapes_ids[ann['category_id']]
            detections.append(
                fo.Detection(label=label, bounding_box=[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
            )
            res = ann['is_valid']
            polyline = fo.Polyline(
                label=label,
                points=[poly],
                closed=True,
                filled=True
            )
            polyline['attributes'] = {'annotation_id': ann['id']}
#             polyline.save()
            polylines.append(polyline)
        tag_is_valid = 'is_valid' if res else 'is_invalid'
        sample = fo.Sample(filepath=filename, id=image_dict['id'], tags=[odm_tag, tag_is_valid])
        sample["ground_truth"] = fo.Detections(detections=detections)
        sample["polylines"] = fo.Polylines(polylines=polylines)
        sample["image_id"] = str(image_dict['id'])
        samples.append(sample)
#     save_json(data, 'dataset/export/fixed_all_json_size.json')

    return samples


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]

def get_fiftyone_dicts(samples):
    samples.compute_metadata()

    dataset_dicts = []
    for idx, sample in enumerate(samples.select_fields(["id", "filepath", "metadata", "polylines"])):
        print(f"Left: {idx}/{len(samples)}", end= '\r')
        if 'is_invalid' in sample.tags: continue
        height = sample.metadata["height"]
        width = sample.metadata["width"]
        
        record = {}
        record["file_name"] = sample.filepath
        record["image_id"] = sample.id
        
        try:
            height, width = cv2.imread(sample.filepath).shape[:2]
        except:
            print("error: ", filename)
            continue

        record["height"] = height
        record["width"] = width
        objs = []
        for fo_poly in sample.polylines['polylines']:
            poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
            poly = [p for x in poly for p in x]
            bbox = bounding_box(fo_poly.points[0])
            bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def get_dataset_dicts_for_correct(img_dir, json_file, state='Train', split=[0.75, 0.25]):
    json_file = json_file
    with open(json_file) as f:
        data = json.load(f)

    bodyes_shapes_ids = {}
    cntt = 0
    for i in data['categories']:
        if i['name'] == 'General body shape':
            cntt += 1
            bodyes_shapes_ids.append({
                int(i['id']): cntt
            })

    skip_data = []
    full_data = {}
    for indices, ann in enumerate(data['annotations']):
        print(f"Left: {len(data['annotations']) - indices} skip: {len(skip_data)}", end='\r')

        if ann['image_id'] in skip_data: continue
        if ann['image_id'] not in full_data:
            image_class = get_image_class(data, ann['image_id'])
            if image_class['fishial_extra']['test_image'] or image_class['fishial_extra']['xray'] or \
                    image_class['fishial_extra']['not_a_real_fish']:
                skip_data.append(ann['image_id'])
                continue
            record = {}
            filename = os.path.join(img_dir, image_class['file_name'])
            try:
                width, height = cv2.imread(filename).shape[:2]
            except:
                print("error: ", filename)
                continue
            record["file_name"] = filename
            record["height"] = width
            record["width"] = height
            record["image_id"] = ann['image_id']
            record["annotations"] = []

            full_data.update({ann['image_id']: record})

        if 'segmentation' in ann and ann['category_id'] in bodyes_shapes_ids:
            if 'skip' in ann:
                print("Skip")
                skip_cnt += 1
                continue

            px = []
            py = []
            for z in range(int(len(ann['segmentation'][0]) / 2)):
                px.append(ann['segmentation'][0][z * 2])
                py.append(ann['segmentation'][0][z * 2 + 1])

            bbox = [np.min(px).tolist(), np.min(py).tolist(), np.max(px).tolist(), np.max(py).tolist()]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann['segmentation'],
                "category_id": 0,  # bodyes_shapes_ids[ann['category_id']],
                "iscrowd": 0
            }

            full_data[ann['image_id']]["annotations"].append(obj)

    cnt_full = sum([len(full_data[z]["annotations"]) for z in full_data])
    cnt_current = 0

    if state == 'Train':
        split[0]
    dataset_dicts = []
    for i in full_data:
        if len(full_data[i]["annotations"]) > 0:
            dataset_dicts.append(full_data[i])

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