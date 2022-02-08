import os
import cv2
import sys
# Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import warnings
warnings.filterwarnings('ignore')
from os import listdir
from os.path import isfile, join
from shapely.geometry import Polygon

from tqdm import tqdm

from shapely.validation import make_valid
from module.classification_package.src.utils import save_json, read_json
from module.segmentation_package.interpreter_segm import SegmentationInference
from module.segmentation_package.src.utils import get_dataset_dicts


def get_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def get_poly_from_coco(polys):
    poly_arrays = []
    for inst in polys:
        poly = inst['segmentation'][0]
        poly_array = []
        for i in range(int(len(poly)/2)):
            poly_array.append( (int(poly[2 * i]), int(poly[2 * i + 1])) )
        poly_arrays.append(make_valid(Polygon(poly_array)))
    return poly_arrays


def get_poly_from_custom(polys, thresholds):
    poly_arrays = []
    for poly_id, poly in enumerate(polys):
        poly_array = []
        for i in range(int(len(poly)/2)):
            poly_array.append( ( int(poly[f"x{i + 1}"]), int(poly[f"y{i + 1}"]) ))
        poly_arrays.append([make_valid(Polygon(poly_array)), False, thresholds[poly_id]])
    return poly_arrays


def get_best_iou(poly, data):
    max_iou = 0
    id_disc = None
    MIN_IOU = 0.22
    for i in range(len(data)):
        intersect = poly.intersection(data[i][0]).area
        union = poly.union(data[i][0]).area
        iou = intersect / union
        if iou > max_iou and iou > MIN_IOU:
            max_iou = iou
            data[i][1] = True
            id_disc = i
    return max_iou, id_disc


input_folder = 'output_aug_custom_schedule_lr'
tmp_folder = os.path.join(input_folder, "score_full")
os.makedirs(tmp_folder, exist_ok=True)

dataset_val = get_dataset_dicts('FishialReaserch/datasets/fishial_collection/cache', "Full",
                  json_file="FishialReaserch/datasets/fishial_collection/export.json")

list_of_files_in_directory = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
array_of_eval_results = []

total_results = []
for file_name in list_of_files_in_directory:
    splited = os.path.splitext(file_name)
    if splited[1] == '.pth':
        json_save_path = os.path.join(tmp_folder, f"{splited[0]}_score.json")
        if f"{splited[0]}_score.json" in [f for f in listdir(tmp_folder) if isfile(join(tmp_folder, f))]:
            data_from_cash = read_json(json_save_path)
            total_results.append([splited[0], data_from_cash])
            print("added: ", splited[0])
            continue
        print("NOW: ", splited[0])
        model_path = os.path.join(input_folder, file_name)
        model_segmentation = SegmentationInference(model_path, device='cuda')
        total_res = {}

        for image_id in tqdm(range(len(dataset_val))):
            image_path = dataset_val[image_id]['file_name']
            annotations = dataset_val[image_id]['annotations']
            img = get_image(image_path)
            array, masks, outputs, thresholds = model_segmentation.simple_inference(img)
            true_poly = get_poly_from_coco(annotations)
            discovered = get_poly_from_custom(array, thresholds)

            dict_with_outcome = {
                'iou': [],
                'threshold': [],
                'area': []
            }

            area_full = dataset_val[image_id]['height'] * dataset_val[image_id]['width']
            for i in range(len(true_poly)):
                iou, id_disc = get_best_iou(true_poly[i], discovered)
                if id_disc:
                    dict_with_outcome['threshold'].append(thresholds[id_disc])
                    dict_with_outcome['area'].append(true_poly[i].area / area_full)
                    dict_with_outcome['iou'].append(iou)
                else:
                    dict_with_outcome['threshold'].append(0)
                    dict_with_outcome['area'].append(true_poly[i].area / area_full)
                    dict_with_outcome['iou'].append(0)

            for ss in discovered:
                if ss[1] == False:
                    dict_with_outcome['threshold'].append(ss[2])
                    dict_with_outcome['area'].append(ss[0].area / area_full)
                    dict_with_outcome['iou'].append(0)
            total_res.update({os.path.basename(image_path): dict_with_outcome})
        save_json(total_res, json_save_path)
        total_results.append([splited[0], total_res])
save_json(total_results, os.path.join(tmp_folder, f"full_score.json"))