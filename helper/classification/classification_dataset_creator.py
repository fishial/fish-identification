import os
import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/fishial/Fishial/Object-Detection-Model')
                
import cv2
import json
import copy
import pandas
import random
import argparse
import requests
import numpy as np
from PIL import Image

from tqdm import tqdm
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

from os import listdir, walk
from os.path import isfile, join
from module.classification_package.src.utils import read_json, save_json
from module.classification_package.src.dataset import FishialDataset
from module.segmentation_package.src.utils import get_mask

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib.pyplot as plt
# import FiftyOne
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

# python /home/fishial/Fishial/Object-Detection-Model/helper/classification/classification_dataset_creator.py -dp /home/fishial/Fishial/dataset/export_07_09_2023/CLASSIFICATION -i /home/fishial/Fishial/dataset/export_07_09_2023/data -a /home/fishial/Fishial/dataset/export_07_09_2023/06_export_Verified_ALL.json -dsn classification-05-09-2023-v0.6 -mei 10 -mpei 0.2 -macipc 350 -micipc 50

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dst_path', type=str, help="The folder when will be saved annotation file and croped images")
    parser.add_argument('-i','--path_to_full_images', type=str, help="The folder when storaged images from coco file")
    
    parser.add_argument('-a','--path_to_src_coco_json', type=str)
    parser.add_argument('-dsn','--voxel_dataset_name', type=str)
    
    parser.add_argument("-mei", "--min_eval_img", type=int, default=10,
                    help="The minimal number of evaluate images in dataset per class")
    parser.add_argument("-mpei", "--max_percent_eva_img", type=float, default=0.2,
                    help="The maximal percent of evaluate images in dataset per class")
    parser.add_argument("-macipc", "--max_cnt_img_per_class", type=int, default=350,
                    help="The maximal number of images per class")
    parser.add_argument("-micipc", "--min_cnt_img", type=int, default=50,
                    help="The maximal number of images per class")
    return parser


def get_category_name(data):
    category_ids = {}
    for idx, i in enumerate(data['categories']):
        if i['name'] == 'General body shape':
            if i['id'] not in category_ids:
                category_ids.update({
                    i['id']: {
                        'name': i['supercategory'],
                        'cnt': 0
                    }
                })
            else:
                category_ids[i['id']]['cnt'] += 1
    return category_ids

def get_category_cnt(data):
    category_cnt = {}
    for i in data['annotations']:
        if 'category_id' in i:
            if i['category_id'] not in category_cnt:
                category_cnt.update({
                    i['category_id']: {
                        'cnt': 0
                    }
                })
            else:
                category_cnt[i['category_id']]['cnt'] += 1
    return category_cnt

def get_class_with_min_ann(data, min_ann = 50):
    data_dict = []
    for i in data:
        if data[i]['cnt'] >= min_ann:
            data_dict.append([i, data[i]['cnt']])
    return data_dict

def find_image_by_id(id: int):
    for i in data['images']:
        if i['id'] == id:
            return i

def get_list_of_files_in_folder(path):
    list_of_files = []
    for (dirpath, dirnames, filenames) in walk(path):
        list_of_files.extend(filenames)
        break
    return list_of_files

def download(url):
    r = requests.get(url[0], allow_redirects=True)  # to get content after redirection
    with open(url[1], 'wb') as f:
        f.write(r.content)
    print("Current: {}".format(url[2]), end='\r')
    
def get_image(data, folder_main, id):
    for img in data['images']:
        if img['id'] == id:
            return cv2.imread(os.path.join(folder_main, img['file_name']))
        
def get_valid_category(data):
    valid_category = {}
    for z in data['categories']:
        if z['name'] == 'General body shape' and z['supercategory'] != 'unknown':
            valid_category.update({z['id']: z['supercategory']})
    return valid_category

def get_all_ann_by_img_id(data_full, img_id, valid_category):
    list_off_ann_for_specific_image = []
    for i in data_full['annotations']:
        try:
            if i['image_id'] == img_id and i['category_id'] in valid_category:
                list_off_ann_for_specific_image.append(i)
        except:
            pass
    return list_off_ann_for_specific_image

def get_mask_by_ann(data, ann, main_folder, box = False):
    polygon_tmp = []
    for pt in range(int(len(ann['segmentation'][0])/2)):
        polygon_tmp.append([int(ann['segmentation'][0][pt * 2]), int(ann['segmentation'][0][pt * 2 + 1])])

    img = get_image(data, main_folder, ann['image_id'])
    if box:
        rect = cv2.boundingRect(np.array(polygon_tmp))
        x, y, w, h = rect
        mask = img[y:y + h, x:x + w].copy()
        if len(mask) == 0:
            return None
    else:
        mask = get_mask(img, np.array(polygon_tmp))
    
    return mask

def fix_poly(poly, shape):
    poly = [ (min(max(0, point[0]), shape[0]), min(max(0, point[1]), shape[1])) for point in poly]
    return poly

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def main(args):
    
    VOXEL_DATASET_NAME = args.voxel_dataset_name

    dst_path = args.dst_path
    path_to_src_coco_json =  args.path_to_src_coco_json
    
    PATH_FULL_IMAGES = args.path_to_full_images

    min_eval_img = args.min_eval_img
    max_percent_eva_img = args.max_percent_eva_img
    max_cnt_img_per_class = args.max_cnt_img_per_class
    min_cnt_img = args.min_cnt_img

    list_of_files = get_list_of_files_in_folder(PATH_FULL_IMAGES)
    data_full = read_json(path_to_src_coco_json)

    folder_to_save_files = os.path.join(dst_path, 'images')
    os.makedirs(folder_to_save_files, exist_ok=True)

    list_of_valid_img = {}
    print(f"MAIN FOLDER PATH: {PATH_FULL_IMAGES}")
    print(f"LIST OF FILES IN MAIN FOLDER: {len(list_of_files)}")

    for i in data_full['images']:
        if 'is_invalid' in i:
            if i['is_invalid']: continue

        if i['file_name'] in list_of_files:
            list_of_valid_img.update({i['id']: i})
    print(f"LIST of IMAGES to USE: {len(list_of_valid_img)}")
    
    valid_category = get_valid_category(data_full)
    ann_per_dict = {valid_category[category]: [] for category in valid_category}

    for ann_id, ann_class in enumerate(data_full['annotations']):
        print(f"Left: {ann_id}/{len(data_full['annotations'])}", end='\r')
        try:
            if not ann_class['image_id'] in list_of_valid_img: continue
            if not ann_class['category_id'] in valid_category: continue
            #if ann_class['is_valid'] == False: continue #used if exist additional anotation flag is_valid which mean annotation was checked in some way 
        except:
            continue

        poly = [(
            int(ann_class['segmentation'][0][point * 2]), 
            int(ann_class['segmentation'][0][point * 2 + 1])) for point in range(int(len(ann_class['segmentation'][0])/2))]
        ann_class['segmentation'] = poly
        ann_class.update({'include_in_odm': list_of_valid_img[ann_class['image_id']]['fishial_extra']['include_in_odm'] })
        ann_per_dict[valid_category[ann_class['category_id']]].append(ann_class)


    for k in list(ann_per_dict):
        if len(ann_per_dict[k]) < min_cnt_img:
            del ann_per_dict[k]
    print(f"TOTAL COUNT OF APPROVED IMAGES: {len(ann_per_dict)}")
    labels_dict = {}
    labels_dict.update({
                label: label_id for label_id, label in enumerate(list(ann_per_dict))
            })
    print(f"LIST OF APPROVED IMAGES: {labels_dict}")

    data_compleated = [[],[],[]]

    for label_name in ann_per_dict:
        ann_per_dict[label_name] = sorted(ann_per_dict[label_name], key=lambda d: d['include_in_odm'], reverse=True) 
        id_end_val = int(max_percent_eva_img * len(ann_per_dict[label_name])) if len(ann_per_dict[label_name]) > int(min_eval_img/max_percent_eva_img) else min_eval_img
        id_end_train = len(ann_per_dict[label_name]) - id_end_val - 1 if len(ann_per_dict[label_name]) - id_end_val <= max_cnt_img_per_class else max_cnt_img_per_class
        data_compleated[0].extend(ann_per_dict[label_name][:id_end_val])
        data_compleated[1].extend(ann_per_dict[label_name][id_end_val:id_end_val + id_end_train])
        data_compleated[2].extend(ann_per_dict[label_name][id_end_val + id_end_train:])

    for data in data_compleated:
        for k in data:
            k.update({'id_internal': labels_dict[valid_category[k['category_id']]]})
            k.update({'label': valid_category[k['category_id']]})

    samples = []
    records = []
    for dataset in zip(data_compleated, ['val', 'train', 'rest']):
        for k_id, ann_inst in enumerate(dataset[0]):
            print(f"dataset: {dataset[1]} | {k_id}/{len(dataset[0])}", end='\r')

            img = cv2.imread(os.path.join(PATH_FULL_IMAGES, list_of_valid_img[ann_inst['image_id']]['file_name']))
            shape = img.shape
            if shape[0] > 2400 or shape[1] > 2400: continue

            ann_inst['segmentation'] = fix_poly(ann_inst['segmentation'], [shape[1], shape[0]])
            rect = cv2.boundingRect(np.array(ann_inst['segmentation']))
            x, y, w, h = rect
            if w < 80 or h < 80: continue

            mask = img[y:y + h, x:x + w]
            ann_inst['segmentation'] = [(v[0] - x, v[1] - y) for v in ann_inst['segmentation']]

            ann_id = ann_inst['id']

            path_to_save = os.path.join(folder_to_save_files, ann_id + ".png")
            try:
                cv2.imwrite(path_to_save, mask)
            except:
                continue

            new_poly = [(z[0]/w, z[1]/h) for z in ann_inst['segmentation']]

            tag_odm = 'odm_true' if ann_inst['include_in_odm'] else 'odm_false'
            records.append(ann_inst)
            sample = fo.Sample(filepath=path_to_save, tags=[dataset[1], tag_odm])
            sample["polyline"] = fo.Polyline(
                                    label=ann_inst['label'],
                                    points=[new_poly],
                                    closed=True,
                                    filled=False)

            sample["area"] = PolyArea([i[0] for i in new_poly],[i[1] for i in new_poly])
            sample['width'] = w
            sample['height'] = h
            sample['drawn_fish_id'] = ann_inst['fishial_extra']['drawn_fish_id']
       
            sample["annotation_id"] = ann_id
            sample["image_id"] = str(ann_inst['image_id'])
            samples.append(sample)
    save_json(records, os.path.join(dst_path, "annotation.json"))


    dataset = fo.Dataset(VOXEL_DATASET_NAME)
    dataset.add_samples(samples)
    dataset.persistent = True
    dataset.save()

if __name__ == '__main__':
    main(arg_parser().parse_args())