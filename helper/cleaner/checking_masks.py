#!/usr/bin/env python3
"""COCO Dataset Viewer.

View images with bboxes from the COCO dataset.
"""
import sys
sys.path.insert(1, '/Users/admin/Desktop/Codahead/Object-Detection-Model')

import argparse
import os
import json
import logging
import urllib.request as ur
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

from module.classification_package.interpreter_classifier import ClassifierFC

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

parser = argparse.ArgumentParser(description="View images with bboxes from the COCO dataset")
parser.add_argument("-a", "--annotations", default='', type=str, metavar="PATH", help="path to annotations json file")
parser.add_argument("-sf", "--folder", default='data/tmp_mask', type=str, metavar="PATH", help="path to folder when will saved fish masks")
parser.add_argument("--data_path", default='', type=str, metavar="PATH", help="path to image_data file")


def read_json(data):
    with open(data) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


class ImageReader:
    def __init__(self, data_local):
        self.data_local = data_local
        self.cash_folder = "data/cache"
        os.makedirs(self.cash_folder, exist_ok=True)
        self.onlyfiles = [f for f in listdir(self.cash_folder) if isfile(join(self.cash_folder, f))]
        print(f"Files {self.onlyfiles}")

    def get_image(self, image):
        return self.__image_reader_local(image['file_name']) if self.data_local else self.__image_reader_url(image['coco_url'], image['file_name'])

    def __image_reader_url(self, url, filename):
        if filename in self.onlyfiles:
            return cv2.imread(os.path.join(self.cash_folder, filename))
        arr = np.asarray(bytearray(ur.urlopen(url).read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
        path = os.path.join(self.cash_folder, filename)
        cv2.imwrite(path, img)
        return img

    def __image_reader_local(self, path):
        img = cv2.imread(os.path.join(self.data_local, path))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def print_info(message: str):
    logging.info(message)


def get_all_annotations(data, image_id, category_list):
    list_ann = []
    try:
        for ann_id, ann in enumerate(data['annotations']):
            if ann['image_id'] == image_id and ann['category_id'] in category_list:
                list_ann.append([ann, ann_id])
    except:
        pass
    return list_ann


def valid_mask(single_mask, valid_cat):
    to_visual = False

    if 'category_id' not in single_mask:
        return to_visual
    else:
        if single_mask['category_id'] not in valid_cat:
            return to_visual

    if len(single_mask['segmentation'][0]) < 20:
        to_visual = True
    return to_visual


def get_correct_category_ids(export):
    categories = []

    for category in export['categories']:
        if category['name'] == 'General body shape':
            categories.append(category['id'])
    return categories


def draw_segm(img, obj):
    color_background = (255, 255, 255)
    color_polyline = (0, 255, 0)
    alpha = 0.5

    polygon = np.array([int(i) for i in obj['segmentation'][0]], dtype=np.int32).reshape((1, int(len(obj['segmentation'][0]) / 2), 2))
    shapes = np.zeros_like(img, np.uint8)
    shapes = cv2.fillPoly(shapes, polygon, color_background, lineType=cv2.LINE_AA)

    mask = shapes.astype(bool)
    out = img.copy()
    out[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]
    out = cv2.polylines(out, polygon, True, color_polyline, 2)
    return out


def draw_bbox(img, obj):
    color = (255, 0, 0)
    x_list = []
    y_list = []

    for i in range(int(len(obj['segmentation'][0]) / 2)):
        x_list.append(int(obj['segmentation'][0][i*2]))
        y_list.append(int(obj['segmentation'][0][i*2 + 1]))
    polygon = np.array([x_list, y_list], dtype=np.int32)
    img = cv2.rectangle(img, (min(polygon[0]), min(polygon[1])),
                    (max(polygon[0]), max(polygon[1])), color, 2)
    return img


def get_invalid(export, category_list):
    mask_to_visual = {}
    for cr_id, image_class in enumerate(export['images']):
        print(f"Left: {len(export['images']) - cr_id}", end='\r')

        for ann_id, ann in enumerate(export['annotations']):
            if ann['image_id'] == image_class['id']:
                infos = valid_mask(ann, category_list)

                if infos:
                    if image_class['id'] in mask_to_visual:
                        mask_to_visual[image_class['id']]['annotations'].append(
                            {
                            'id': ann_id,
                            'object': ann
                        })
                    else:
                        mask_to_visual.update({
                            image_class['id']:{
                                'annotations': [
                                    {
                                        'id': ann_id,
                                        'object': ann,
                                    }
                                ],
                                'image': {
                                    'coco_url': image_class['coco_url'],
                                    'file_name': image_class['file_name']
                                }
                            }
                        })
    return mask_to_visual


def get_mask(image, segm):
    polygon = []

    for i in range(int(len(segm) / 2)):
        polygon.append([int(segm[2 * i]), int(segm[2 * i + 1])])

    pts = np.array(polygon)
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


def main():
    print_info("Starting...")
    args = parser.parse_args()

    model_finder = ClassifierFC('data/best_score.ckpt', n_classes=2)

    main_folder = args.folder
    os.makedirs(os.path.join(main_folder, "0"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "1"), exist_ok=True)
    os.makedirs(os.path.join(main_folder, "2"), exist_ok=True)

    export_dict = read_json(args.annotations)
    img_reader = ImageReader(args.data_path)

    category_list = get_correct_category_ids(export_dict)
    # mask_to_visual = get_invalid(export_dict, category_list)
    # save_json(mask_to_visual, "test_saver.json")
    mask_to_visual = read_json("data/test_saver.json")
    for idx, image_class in enumerate(mask_to_visual):
        print(f"Left: {len(mask_to_visual) - idx}")
        img = img_reader.get_image(mask_to_visual[image_class]['image'])
        for ann in mask_to_visual[image_class]['annotations']:
            try:
                cuted_img = get_mask(img, ann['object']['segmentation'][0])
                output = model_finder.inference_numpy(cuted_img, top_k=2)
                print(output)
                if output[0][0]:
                    path = os.path.join(os.path.join(main_folder, "0"), f"{ann['object']['image_id']}_{ann['id']}.png")
                else:
                    path = os.path.join(os.path.join(main_folder, "1"), f"{ann['object']['image_id']}_{ann['id']}.png")
                cv2.imwrite(path, cuted_img)
            except Exception as e:
                print(f"Error: {e}")
                pass

    # for idx, image_class in enumerate(export_dict['images']):
    #     print(f"Left: {len(export_dict['images']) - idx}")
    #     list_ann = get_all_annotations(export_dict, image_class['id'], category_list)
    #     img = img_reader.get_image(image_class)
    #     for ann in list_ann:
    #         try:
    #             cuted_img = get_mask(img, ann[0]['segmentation'][0])
    #             output = model_finder.inference_numpy(cuted_img, top_k=2)
    #             if not output[0][0]:
    #                 path = os.path.join(os.path.join(main_folder, "2"), f"{ann[0]['image_id']}_{ann[1]}.png")
    #                 cv2.imwrite(path, cuted_img)
    #         except Exception as e:
    #             print(f"Error: {e}")
    #             pass

    print(f"annotations: {args.annotations} \ndata_path: |{args.data_path}|")


if __name__ == "__main__":
    main()
