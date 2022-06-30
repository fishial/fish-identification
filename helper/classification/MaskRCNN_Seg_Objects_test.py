import sys

# Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')
import os
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from module.segmentation_package.interpreter_segm import SegmentationInference
from module.segmentation_package.src.utils import get_dataset_dicts, get_prepared_data
from module.classification_package.src.utils import read_json, save_json
import cv2

import warnings

warnings.filterwarnings('ignore')

from os import listdir
from os.path import isfile, join
from matplotlib.pyplot import figure


def get_iou_poly(polygon1, polygon2):
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou


figure(figsize=(10, 10), dpi=200)
plt.rcParams["figure.figsize"] = (20, 20)

input_folder = '../../output/segmentation/amp_on_new_ds/18_06_2022_06_55_38'

tmp_data, labels_list = get_prepared_data(
    '/home/codahead/Fishial/FishialReaserch/datasets/fishial_collection_V2.0/FULL',
    '/home/codahead/Fishial/new_data_set/export_Verified_ALL_v2.json',
    r'/home/codahead/Fishial/FishialReaserch/output/train_results/resnet_18_98_finall_update/train+test_labels.json')

data_set = get_dataset_dicts(tmp_data, 'Test')

list_of_files_in_directory = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]

for file_name in list_of_files_in_directory:
    splited = os.path.splitext(file_name)
    if splited[1] != '.pth': continue

    model_segmentation = SegmentationInference(os.path.join(input_folder, file_name), device='cuda', class_num=1,
                                               labels_list=['Fish'])
    model_segmentation.re_init_model(0.1)

    true_dict = {
        'FP': [],
        'SIZE': [],
        'STATE': []
    }

    for rec_id in range(len(data_set)):
        print(f"Left: {len(data_set) - rec_id}", end='\r')
        poly_list = []
        for poly_single in data_set[rec_id]['annotations']:
            poly_tmp = poly_single['segmentation'][0]
            poly_tmp_shapely = Polygon([(int(poly_tmp[2 * point_id]), int(poly_tmp[2 * point_id + 1])) for point_id in
                                        range(int(len(poly_tmp) / 2))])
            poly_list.append(poly_tmp_shapely)

        img = cv2.imread(data_set[rec_id]['file_name'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meta_data, masks, outputs = model_segmentation.simple_inference(img)
        poly_seg_list = []
        for poly_seg in meta_data['polygons']:
            poly_seg_shapely = Polygon(
                [(int(poly_seg[f"x{point_id + 1}"]), int(poly_seg[f"y{point_id + 1}"])) for point_id in
                 range(int(len(poly_seg) / 2))])
            poly_seg_list.append(poly_seg_shapely)

        for true_poly_id, true_poly in enumerate(poly_list):
            true_dict['SIZE'].append(true_poly.area)
            true_dict['STATE'].append([])

            state = 0.0
            list_of_used = []
            for found_poly_id in range(len(poly_seg_list)):
                try:
                    iou = get_iou_poly(true_poly, poly_seg_list[found_poly_id])
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                if iou < 0.1: continue
                list_of_used.append(found_poly_id)
                true_dict['STATE'][len(true_dict['STATE']) - 1].append(
                    [round(iou, 2), round(meta_data['scores'][found_poly_id], 2)])
            list_of_scores_ununsed = set(range(0, len(poly_seg_list))) - set(list_of_used)
            true_dict['FP'].extend(
                [round(meta_data['scores'][found_poly_id], 2) for found_poly_id in list_of_scores_ununsed])

    save_json(true_dict, os.path.join(input_folder, f"{splited[0]}_eval_test.json"))
