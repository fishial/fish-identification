import argparse
import json 
import cv2

import os
from os import listdir, walk
from os.path import isfile, join

import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
def read_json(data):
    with open(data) as f:
        return json.load(f)
    
def get_list_of_files_in_folder(path):
    list_of_files = []
    for (dirpath, dirnames, filenames) in walk(path):
        list_of_files.extend(filenames)
        break
    return list_of_files

def check_key(instance, key):
    if key in instance:
        return instance[key]
    else:
        return True
    
def fix_poly(poly, shape):
    poly = [ (min(max(0, point[0]), shape[1]), min(max(0, point[1]), shape[1])) for point in poly]
    return poly

def PolyArea(poly):
    x = []
    y = []
    
    for point in poly:
        x.append(point[0])
        y.append(point[1])
        
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_min_area(sample):
    min_area = 1.1
    for z in sample:
        area = PolyArea(z.points[0])
        if min_area > area:
            min_area = area 
    return min_area

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coco_path', type=str)
    parser.add_argument('-i','--image_path', type=str)
    parser.add_argument('-ds','--data_set', type=str)

    return parser

#python converterCocoToVoxel.py -c '/home/fishial/Fishial/dataset/export/03_export_Verified_ALL.json.json' -i '/home/fishial/Fishial/dataset/fishial_collection/data' -ds 'export-fishial-november-test'
def main(args):
   
    COCO_PATH = args.coco_path  #r'/home/fishial/Fishial/dataset/export/03_export_Verified_ALL.json.json'
    IMAGE_FOLDER = args.image_path #"/home/fishial/Fishial/dataset/fishial_collection/data"
    DATASET_NAME = args.data_set

    data_coco_export = read_json(COCO_PATH)

    categories = {}
    for k in data_coco_export['categories']:
        categories.update({k['id']: {
            'name':k['name'],
            'supercategory':k['supercategory'] 
        }})

    list_of_files = get_list_of_files_in_folder(IMAGE_FOLDER)
    print(IMAGE_FOLDER)
    print(len(list_of_files))
    
    images_infos = {}
    for inst_id, image_inst in enumerate(data_coco_export['images']):
        
        print(f"Left: {inst_id}/{len(data_coco_export['images'])} count: {len(images_infos)}", end='\r')
        
        file_name = image_inst['file_name']
        if file_name not in list_of_files: continue
        if 'fishial_extra' not in image_inst: continue
        if check_key(image_inst['fishial_extra'], 'xray'): continue
        if check_key(image_inst['fishial_extra'], 'not_a_real_fish'): continue
        if check_key(image_inst['fishial_extra'], 'no_fish'): continue
        if check_key(image_inst['fishial_extra'], 'test_image'): continue

        try:
            image = cv2.imread(os.path.join(IMAGE_FOLDER, file_name))
        except Exception as e:
            continue
        height, width = image.shape[:2]
        images_infos.update({
            image_inst['id']: {
                'file_name': os.path.join(IMAGE_FOLDER, file_name),
                'width': width,
                'height': height,
                'include_in_odm': image_inst['fishial_extra']['include_in_odm']
            }
        })
    print(f"LENGHT1: {len(images_infos)}                                 ")
    print(f"LENGHT2: {len(data_coco_export['annotations'])}")
    annotations_infos = {}
    for ann_id, ann_inst in enumerate(data_coco_export['annotations']):
        print(f"Left: {ann_id}/{len(data_coco_export['annotations'])}", end='\r')

        image_id = ann_inst['image_id']

        try:
            drawn_fish_id = ann_inst['fishial_extra']['drawn_fish_id']

            poly = [(
                int(ann_inst['segmentation'][0][point * 2]), 
                int(ann_inst['segmentation'][0][point * 2 + 1])) for point in range(int(len(ann_inst['segmentation'][0])/2))]
            poly = fix_poly(poly, [images_infos[image_id]['height'], images_infos[image_id]['width']])
            poly = [[points[0]/images_infos[image_id]['width'], points[1]/images_infos[image_id]['height']] for points in poly]
        except Exception as e:
            continue
        ann_id = ann_inst['id']
        category_id = ann_inst['category_id']
        if 'poly' in images_infos[image_id]:
            images_infos[image_id]['poly'].append([categories[category_id],drawn_fish_id,ann_id, poly])
        else:
            images_infos[image_id].update({
            'poly': [[categories[category_id],drawn_fish_id,ann_id, poly]]
        })

    samples = []
    for idx, image_id in enumerate(images_infos):
        print(f"Left: {idx}/{len(images_infos)}", end='\r')
        if 'poly' not in images_infos[image_id]: continue
        if len(images_infos[image_id]['poly']) == 0: continue 

        polylis = []
        for poly in images_infos[image_id]['poly']:
            category, drawn_fish_id, ann_id, polylines = poly
            polyline = fo.Polyline(
                    label=category['supercategory'],
                    points=[polylines],
                    closed=True,
                    filled=False)
            polyline['tags'] = [category['name']]
            polyline['ann_id'] = ann_id
            polyline['drawn_fish_id'] = drawn_fish_id
            polylis.append(polyline)
            
        odm_tag = 'include_in_odm' if images_infos[image_id]['include_in_odm'] else 'not_include_in_odm'
        sample = fo.Sample(filepath=images_infos[image_id]['file_name'], tags=[odm_tag])
        sample["polylines"] = fo.Polylines(polylines=polylis)
        sample["image_id"] = image_id
        sample["min_area"] = 100 * get_min_area(polylis)
        sample["width"] = images_infos[image_id]['width']
        sample["height"] = images_infos[image_id]['height']

        samples.append(sample)

    dataset = fo.Dataset(DATASET_NAME)
    dataset.persistent = True
    dataset.add_samples(samples)

    print("Finish")

if __name__ == '__main__':
    main(arg_parser().parse_args())