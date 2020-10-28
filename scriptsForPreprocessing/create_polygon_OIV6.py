import json
import numpy as np
from pycocotools import mask
from skimage import measure
import os, json
import cv2
import pandas as pd
from shutil import copyfile
from imantics import Polygons, Mask


fishial_dataset = r'resources/val'
os.makedirs(fishial_dataset, exist_ok=True)

mask_dir = r'resources/val-fish-mask'
img_dir = r'resources/val-fish-img'
path_to_csv = r'resources/validation-annotations-object-segmentation-fish.csv'

list_path_img = os.listdir(img_dir)
list_path_mask = os.listdir(mask_dir)

df = pd.read_csv(path_to_csv)

count = 0
result_dict = {}
for index, row in df.iterrows():
    print("Score: ", len(df) - index, len(result_dict))
    if row['MaskPath'] in list_path_mask and row['ImageID']+".jpg" in list_path_img:
        fullname_mask = os.path.join(mask_dir, row['MaskPath'])
        fullname_image = os.path.join(img_dir, row['ImageID']+".jpg")
        ground_truth_binary_mask = cv2.imread(fullname_mask, 0)
        img_tmp = cv2.imread(fullname_image)
        w, h, _ = img_tmp.shape
        ground_truth_binary_mask = cv2.resize(ground_truth_binary_mask, (h, w))
        title, ext = os.path.splitext(os.path.basename(fullname_mask))
        ground_truth_binary_mask_zero = np.zeros(ground_truth_binary_mask.shape)
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)
        if len(contours) < 1: continue

        B = None
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            B = np.reshape(segmentation, (-1, 2))
            if len(B) > 20: break

        if B is None: continue

        x_array = []
        y_array = []

        for i in B:
            x_array.append(int(i[0]))
            y_array.append(int(i[1]))
        #     cv2.circle(img_tmp, (int(i[0]), int(i[1])), 2, (0, 255, 0), -1)
        # cv2.imshow('image', img_tmp)
        # cv2.waitKey(0)

        if len(x_array) < 20 or len(y_array) < 20:
            print("--------------------------------------------")
            print("x_array: ", len(x_array))
            print("y_array: ", len(y_array))
            continue


        # polygons = Mask(ground_truth_binary_mask).polygons()
        # x_array = []
        # y_array = []
        # for i in polygons.points[0]:
        #     x_array.append(int(i[0]))
        #     y_array.append(int(i[1]))
        #     cv2.circle(img_tmp, (int(i[0]), int(i[1])), 2, (0, 255, 0), -1)
        # cv2.imshow('image', img_tmp)
        # cv2.waitKey(0)
        # if len(x_array) < 8: continue
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