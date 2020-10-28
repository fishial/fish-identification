import json
import numpy as np
from pycocotools import mask
from skimage import measure
import os, json
import cv2
import torch, torchvision

path_new_dataset = r'path/to/dataset'
os.makedirs(path_new_dataset, exist_ok=True)

for folder_check, count_img in zip(['val', 'train'], (30, 200)):
    current_folder_new = os.path.join(path_new_dataset, folder_check)
    os.makedirs(current_folder_new, exist_ok=True)

    marker_path = os.path.join(r'folder_with_data', folder_check)
    file_name = r'via_region_data.json'

    path_to_file = os.path.join(marker_path, file_name)
    with open(path_to_file) as f:
        json1_data = json.load(f)
    result_dict = {}

    list_path = os.listdir(marker_path)
    title_on_json = [i for i in json1_data['_via_img_metadata']]

    for idx, i in enumerate(json1_data['_via_img_metadata']):
        # if count_img < len(result_dict): continue
        path_to_image = os.path.join(marker_path, i)
        img = cv2.imread(path_to_image)
        w, h = img.shape[:2]

        # if h > 1440 or w > 1920: continue

        title, ext = os.path.splitext(os.path.basename(os.path.join(marker_path, i)))
        print(os.path.join(marker_path, i), img.shape[:2], len(result_dict), title, ext)
        new_title = title + ".jpg"

        cv2.imwrite(os.path.join(current_folder_new, new_title), img)
        all_points_x = json1_data['_via_img_metadata'][i]['regions'][0]['shape_attributes']['all_points_x']
        all_points_y = json1_data['_via_img_metadata'][i]['regions'][0]['shape_attributes']['all_points_y']

        result_dict.update({
            new_title + str(w * h): {
                "fileref": "",
                "size": w * h,
                "filename": new_title,
                "base64_img_data": "",
                "file_attributes": {

                },
                "regions": {
                    "0": {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": all_points_x,
                            "all_points_y": all_points_y
                        },
                        "region_attributes": {

                        }
                    }
                }
            }
        })


    with open(os.path.join(current_folder_new, file_name), 'w') as fp:
        json.dump(result_dict, fp)