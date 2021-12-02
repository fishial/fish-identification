import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import os
import json
import cv2
import numpy as np

from module.segmentation_package.src.utils import get_mask


# This script will cut objects from the COCO fish export file, 
# create a folder and place all the cut images in a specific folder that corresponds 
# to the class id from the COCO file, so you need to specify the new folder name 
# and path to the COCO json file.
def get_image(data, folder_main, id):
    for img in data['images']:
        if img['id'] == id:
            try:
                state = "Train" if img['train_data'] else "Test"
                path_img = os.path.join(folder_main, state)
                return cv2.imread(os.path.join(path_img, img['file_name']))
            except:
                return None

#destination folder
folder_name = "dataset"
# path to coco file 
json_file = "fishial_collection/fishial_collection_correct.json"

with open(json_file) as f:
    data = json.load(f)

categories_body = []
for z in data['categories']:
    if z['name'] == 'General body shape':
        categories_body.append(z['id'])

for idx, z in enumerate(data['annotations']):
    if not 'category_id' in z:
        continue
        
    print("Left: {}".format(len(data['annotations']) - idx), end='\r')

    if z['category_id'] in categories_body:
        tmp_folder = os.path.join(folder_name, str(z['category_id']))
        os.makedirs(tmp_folder, exist_ok=True)
        
        polygon_tmp = []

        for pt in range(int(len(z['segmentation'][0])/2)):
            polygon_tmp.append([int(z['segmentation'][0][pt * 2]), int(z['segmentation'][0][pt * 2 + 1])])

        img = get_image(data, "fishial_collection", z['image_id'])
        if img is None:
            continue
        mask = get_mask(img, np.array(polygon_tmp))
        try:
            cv2.imwrite(os.path.join(tmp_folder, "image_{}.png".format(z['image_id'])), mask)
        except:
            print("polygon_tmp: {}".format(len(polygon_tmp)))
