import json
import numpy as np
from pycocotools import mask
from skimage import measure
import os, json
import cv2
import pandas as pd
from shutil import copyfile
import re
from imantics import Polygons, Mask

fishial_dataset = r'resources/new_part'
os.makedirs(fishial_dataset, exist_ok=True)

mask_dir = r'resources/old_data/train_label'
img_dir = r'resources/old_data/train'

list_path_img = os.listdir(img_dir)
list_path_mask = os.listdir(mask_dir)

count = 0
result_dict = {}
list_png = os.listdir(mask_dir)
for index, mask_path in enumerate(list_png):
    title, ext = os.path.splitext(os.path.basename(mask_path))
    match = re.finditer("([_]{1})", title)
    base_title = ""
    for i in match:
        base_title = title[:i.start(0)]
    if mask_path in list_path_mask and base_title + ".jpeg" in list_path_img:
        fullname_mask = os.path.join(mask_dir, mask_path)
        fullname_image = os.path.join(img_dir, base_title + ".jpeg")
        ground_truth_binary_mask = cv2.imread(fullname_mask, 0)

        img_tmp = cv2.imread(fullname_image)
        w, h, _ = img_tmp.shape
        ground_truth_binary_mask = cv2.resize(ground_truth_binary_mask, (h, w))

        new_size_w = int(w * 0.03)
        new_size_h = int(h * 0.03)

        ground_truth_binary_mask = ground_truth_binary_mask[new_size_w: w - new_size_w, new_size_h: h - new_size_h]
        img_tmp = img_tmp[new_size_w: w - new_size_w, new_size_h: h - new_size_h]
        polygons = Mask(ground_truth_binary_mask).polygons()

        title, ext = os.path.splitext(os.path.basename(fullname_mask))
        x_array = []
        y_array = []
        for i in polygons.points[0]:
            x_array.append(int(i[0]))
            y_array.append(int(i[1]))
        if len(x_array) < 8: continue

        # cv2.circle(img_tmp, (int(i[0]), int(i[1])), 2, (0, 255, 0), -1)
        # cv2.imshow('image', img_tmp)
        # cv2.waitKey(0)

        dst_path = os.path.join(fishial_dataset, os.path.basename(fullname_image))
        copyfile(fullname_image, dst_path)
        result_dict.update({
            title: {
                "fileref": "",
                "size": w*h,
                "filename": os.path.basename(fullname_image),
                "base64_img_data": "",
                "file_attributes": {
                },
                "regions": {
                    "0": {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": x_array,
                            "all_points_y": y_array
                        },
                        "region_attributes": {

                        }
                    }
                }
            }
        })

with open(os.path.join(fishial_dataset, 'via_region_data.json'), 'w') as fp:
    json.dump(result_dict, fp)
