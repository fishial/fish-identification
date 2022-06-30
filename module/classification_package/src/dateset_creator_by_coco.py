import os
import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')
                
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


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coco_path', type=str)
    parser.add_argument('-f','--dst_path', type=str)
    parser.add_argument('-sp','--src_path', type=str)
    parser.add_argument('-ec','--min_eval_count', type=int, default=15)
    parser.add_argument('-ep','--min_eval_percent', type=float, default=0.2)
    parser.add_argument('-mc','--max_cnt_per_class', type=int, default=211)
    parser.add_argument('-nc','--num_classes', type=int, default=75)
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

def get_mask_by_ann(data, ann, main_folder):
    polygon_tmp = []
    for pt in range(int(len(ann['segmentation'][0])/2)):
        polygon_tmp.append([int(ann['segmentation'][0][pt * 2]), int(ann['segmentation'][0][pt * 2 + 1])])

    img = get_image(data, main_folder, ann['image_id'])
    mask = get_mask(img, np.array(polygon_tmp))
    
    return mask


def main(args):
    dst_path = args.dst_path
    path_to_src_coco_json =  args.coco_path #'../../new_data_set/export_Verified_ALL.json'
    
    path_to_new_dataset = args.src_path

    min_eval_img = args.min_eval_count
    max_percent_eva_img = args.min_eval_percent
    max_cnt_img_per_class = args.max_cnt_per_class
    num_classes = args.num_classes
    
    list_of_files = get_list_of_files_in_folder(dst_path)
    data_full = read_json(path_to_src_coco_json)

    list_sd = []
    urls = []
    for i in range(len(data_full['images'])):
        list_sd.append(data_full['images'][i]['file_name'])
        path = os.path.join(dst_path, data_full['images'][i]['file_name'])
        if os.path.basename(path) not in list_of_files:
            urls.append([data_full['images'][i]['coco_url'], path, i])
    print(f'Count image left to downloand {len(urls)}')
    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(download, urls) #urls=[list of url]

    list_imgs_odm_first_priority  = []
    list_imgs_odm_second_priority = []
    list_of_files = get_list_of_files_in_folder(dst_path)

    for i in data_full['images']:
        if i['fishial_extra']['include_in_odm'] and i['file_name'] in list_of_files:
            list_imgs_odm_first_priority.append([i['id'], i['file_name']])
        elif not i['fishial_extra']['include_in_odm'] and i['file_name'] in list_of_files:
            list_imgs_odm_second_priority.append([i['id'], i['file_name']])

    valid_category = get_valid_category(data_full)

    dict_cnt_ann_per_sepecific_class_first_priority = {}
    for image_id in list_imgs_odm_first_priority:
        list_of_ann = get_all_ann_by_img_id(data_full, image_id[0], valid_category)
        for ann_for_img in list_of_ann:

            if not ann_for_img['category_id'] in valid_category:
                continue

            if ann_for_img['category_id'] in dict_cnt_ann_per_sepecific_class_first_priority:
                dict_cnt_ann_per_sepecific_class_first_priority[ann_for_img['category_id']].append(ann_for_img)
            else:
                dict_cnt_ann_per_sepecific_class_first_priority.update({ann_for_img['category_id']: [ann_for_img]})

    dict_cnt_ann_per_sepecific_class_second_priority = {}
    for image_id in list_imgs_odm_second_priority:
        list_of_ann = get_all_ann_by_img_id(data_full, image_id[0], valid_category)
        for ann_for_img in list_of_ann:

            if not ann_for_img['category_id'] in valid_category:
                continue

            if ann_for_img['category_id'] in dict_cnt_ann_per_sepecific_class_second_priority:
                dict_cnt_ann_per_sepecific_class_second_priority[ann_for_img['category_id']].append(ann_for_img)
            else:
                dict_cnt_ann_per_sepecific_class_second_priority.update({ann_for_img['category_id']: [ann_for_img]})

    list_all = [[i, len(dict_cnt_ann_per_sepecific_class_first_priority[i])] 
                for i in dict_cnt_ann_per_sepecific_class_first_priority]
    list_all = sorted(list_all, key=lambda x: x[1], reverse=True)

    data_json_test  = {'image_id': [], 'label_encoded': [], 'label': [], 'img_path': [] }
    data_json_train = {'image_id': [], 'label_encoded': [], 'label': [], 'img_path': [] }

    global_counter = 0

    for left, counter in enumerate(list_all[:num_classes]):
        class_id = counter[0]
        full_list = copy.deepcopy(dict_cnt_ann_per_sepecific_class_first_priority[class_id])
        if class_id in dict_cnt_ann_per_sepecific_class_second_priority:
            full_list.extend(dict_cnt_ann_per_sepecific_class_second_priority[class_id])
        total_cnt = min(max_cnt_img_per_class, len(full_list))
        eval_cnt = max(min_eval_img, int(total_cnt * max_percent_eva_img))

        path_to_specific_class_train = os.path.join(os.path.join(path_to_new_dataset, 'train'), str(class_id))
        path_to_specific_class_test = os.path.join(os.path.join(path_to_new_dataset, 'test'), str(class_id))
        os.makedirs(path_to_specific_class_train, exist_ok=True)
        os.makedirs(path_to_specific_class_test, exist_ok=True)
        
        small_counter = 0
        for indieces in full_list:

            if small_counter >= total_cnt: continue

            print(f'Idx: {left}/{num_classes} | {small_counter} - {eval_cnt}/{total_cnt}/{len(full_list)} | Current class: {valid_category[class_id]} | num: {global_counter} {10 * " "}', end = '\r')

            mask = get_mask_by_ann(data_full, indieces, dst_path)

            if mask is None: continue

            img_name = "{}.png".format(indieces['id'])
            
            if small_counter < eval_cnt:
                path_to_cut_img = os.path.join(path_to_specific_class_test, img_name)
                img_path = os.path.join('test', os.path.join(str(class_id), img_name))
                cv2.imwrite(path_to_cut_img, mask)
                  
                data_json_test['image_id'].append(indieces['id'])
                data_json_test['label_encoded'].append(class_id)
                data_json_test['label'].append(valid_category[class_id])
                data_json_test['img_path'].append(img_path)
            else:
                path_to_cut_img = os.path.join(path_to_specific_class_train, img_name)
                img_path = os.path.join('train', os.path.join(str(class_id), img_name))
                cv2.imwrite(path_to_cut_img, mask)
                  
                data_json_train['image_id'].append(indieces['id'])
                data_json_train['label_encoded'].append(class_id)
                data_json_train['label'].append(valid_category[class_id])
                data_json_train['img_path'].append(img_path)

            global_counter += 1
            small_counter  += 1

    save_json(data_json_test, os.path.join(path_to_new_dataset,'data_test.json'))
    save_json(data_json_train, os.path.join(path_to_new_dataset,'data_train.json'))

    data_remaining  = {'image_id': [], 'label_encoded': [], 'label': [], 'img_path': [] }

    for left, counter in enumerate(list_all[:num_classes]):
        class_id = counter[0]

        full_list = copy.deepcopy(dict_cnt_ann_per_sepecific_class_first_priority[class_id])
        if class_id in dict_cnt_ann_per_sepecific_class_second_priority:
            full_list.extend(dict_cnt_ann_per_sepecific_class_second_priority[class_id])
        total_cnt = min(max_cnt_img_per_class, len(full_list))

        path_to_specific_class = os.path.join(os.path.join(path_to_new_dataset, 'remain'), str(class_id))
        os.makedirs(path_to_specific_class, exist_ok=True)
        small_counter = 0
        for indieces in full_list:

            cnt_done = len(data_remaining['image_id'])
            print(f'Idx: {left}/{num_classes} | {len(full_list)} | Current class: {valid_category[class_id]} | num: {cnt_done}/{global_counter} {10 * " "}', end = '\r')
            mask = get_mask_by_ann(data_full, indieces, dst_path)
            if mask is None: continue
            global_counter += 1
            small_counter  += 1

            if small_counter < total_cnt: continue
                  
            img_name = "{}.png".format(indieces['id'])

            path_to_cut_img = os.path.join(path_to_specific_class, img_name)
            img_path = os.path.join('remain', os.path.join(str(class_id), img_name))

            cv2.imwrite(path_to_cut_img, mask)

            data_remaining['image_id'].append(indieces['id'])
            data_remaining['label_encoded'].append(class_id)
            data_remaining['label'].append(valid_category[class_id])
            data_remaining['img_path'].append(img_path)

    save_json(data_remaining, os.path.join(path_to_new_dataset,'data_remain.json'))
                  
    data_out_of_class  = {'image_id': [], 'label_encoded': [], 'label': [], 'img_path': [] }
    
    left_classes = len(list_all[num_classes:])
    for left, counter in enumerate(list_all[num_classes:]):
        class_id = counter[0]

        full_list = copy.deepcopy(dict_cnt_ann_per_sepecific_class_first_priority[class_id])
        if class_id in dict_cnt_ann_per_sepecific_class_second_priority:
            full_list.extend(dict_cnt_ann_per_sepecific_class_second_priority[class_id])
        total_cnt = min(max_cnt_img_per_class, len(full_list))

        path_to_specific_class = os.path.join(os.path.join(path_to_new_dataset, 'out_of_class'), str(class_id))
        os.makedirs(path_to_specific_class, exist_ok=True)
        for indieces in full_list:

            cnt_done = len(data_out_of_class['image_id'])
            print(f'Idx: {left}/{left_classes} | {len(full_list)} | Current class: {valid_category[class_id]} | num: {cnt_done}/{global_counter} {10 * " "}', end = '\r')
            mask = get_mask_by_ann(data_full, indieces, dst_path)
            if mask is None: continue
            global_counter += 1
                  
            img_name = "{}.png".format(indieces['id'])

            path_to_cut_img = os.path.join(path_to_specific_class, img_name)
            img_path = os.path.join('out_of_class', os.path.join(str(class_id), img_name))

            cv2.imwrite(path_to_cut_img, mask)
                  
            data_out_of_class['image_id'].append(indieces['id'])
            data_out_of_class['label_encoded'].append(class_id)
            data_out_of_class['label'].append(valid_category[class_id])
            data_out_of_class['img_path'].append(img_path)

    save_json(data_out_of_class, os.path.join(path_to_new_dataset,'data_out_of_class.json'))
              
              
if __name__ == '__main__':
    main(arg_parser().parse_args())