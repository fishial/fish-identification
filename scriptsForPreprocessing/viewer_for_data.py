import os, json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


# import data
path_to_json = r'resources/parsing/via_region_data.json'

# path to augmented dataset
path_to_aug_dataset = r'resources/parsing'
json_tmp = json.load(open(path_to_json))
unique_img_array = []

while len(json_tmp) != 0:
    keys = list(json_tmp.keys())
    single_img = [json_tmp[keys[len(keys) - 1]]]
    img_name = json_tmp[keys[len(keys) - 1]]['filename']
    del json_tmp[keys[len(keys) - 1]]
    for idx in range(len(json_tmp) - 1, -1, -1):
        if json_tmp[keys[idx]]['filename'] == img_name:
            single_img.append(json_tmp[keys[idx]])
            del json_tmp[keys[idx]]
    unique_img_array.append(single_img)

# create folder for augumented dataset !
os.makedirs(path_to_aug_dataset, exist_ok=True)
result_dict = {}

for leave, i in enumerate(unique_img_array):
    print("Score: ", len(unique_img_array) - leave, len(result_dict))
    img_main = os.path.join(path_to_aug_dataset, i[0]['filename'])
    image = cv2.imread(img_main)
    h, w, _ = image.shape

    for idx_, i_idx in enumerate(i):
        img_test = image.copy()
        color = random_color()
        polygon_calc = []
        for polygon_idx in range(len(i_idx['regions']['0']['shape_attributes']['all_points_x'])):
            polygon_calc.append((i_idx['regions']['0']['shape_attributes']['all_points_x'][polygon_idx],
                                 i_idx['regions']['0']['shape_attributes']['all_points_y'][polygon_idx]))
        if len(polygon_calc) < 8: continue
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(list(map(tuple, polygon_calc)), outline=1, fill=255)
        mask = np.array(img)
        mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

        dst = cv2.addWeighted(image, 0.6, mask_stack, 0.2, 0)
        color = (0, 0, 255)
        if i_idx['correct']:
            color = (0, 255, 0)

        cv2.rectangle(dst, (min(i_idx['regions']['0']['shape_attributes']['all_points_x']),
                            min(i_idx['regions']['0']['shape_attributes']['all_points_y'])),
                      (max(i_idx['regions']['0']['shape_attributes']['all_points_x']),
                       max(i_idx['regions']['0']['shape_attributes']['all_points_y'])), color, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(dst, 'Mask: {}/{} \n Verified {} Correcr {} '.format(idx_ + 1, len(i), i_idx['verified'], i_idx['correct']), (10, 50),
                    font, 1, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('img', dst)  # Display
        cv2.waitKey()
