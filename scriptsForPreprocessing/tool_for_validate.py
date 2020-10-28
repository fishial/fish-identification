import os, json
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


# import data
path_to_json = r'resources/fish-img-unique/via_region_data.json'
# path to augmented dataset
path_to_aug_dataset = r'resources/fish-img-unique'
json_tmp = json.load(open(path_to_json))
unique_img_array = []
json_tmp_copy = json_tmp.copy()

while len(json_tmp) != 0:
    keys = list(json_tmp.keys())
    single_img = [[json_tmp[keys[len(keys) - 1]], keys[len(keys) - 1]]]

    img_name = json_tmp[keys[len(keys) - 1]]['filename']
    del json_tmp[keys[len(keys) - 1]]
    for idx in range(len(json_tmp) - 1, -1, -1):
        if json_tmp[keys[idx]]['filename'] == img_name:
            single_img.append([json_tmp[keys[idx]], keys[idx]])
            del json_tmp[keys[idx]]
    unique_img_array.append(single_img)

# create folder for augumented dataset !
os.makedirs(path_to_aug_dataset, exist_ok=True)
cnt = 0
for leave, i in enumerate(unique_img_array):
    print("===============You have to process {} image: left mask to verify {} cnt: {} =============== ".format(len(unique_img_array) - leave,
                                                                                                                len(json_tmp_copy) - cnt, cnt))
    img_main = os.path.join(path_to_aug_dataset, i[0][0]['filename'])
    image = None

    for idx_, polygon_cls in enumerate(i):
        cnt +=1
        title = polygon_cls[1]  # title of dictionary
        i_idx = polygon_cls[0] # take dictionary
        if i_idx['verified']: continue
        if image is None:
            image = cv2.imread(img_main)
        h, w, _ = image.shape
        img_test = image.copy()
        color = random_color()
        polygon_calc = []
        for idx_polygon in range(len(i_idx['regions']['0']['shape_attributes']['all_points_x'])):
            polygon_calc.append((i_idx['regions']['0']['shape_attributes']['all_points_x'][idx_polygon],
                                 i_idx['regions']['0']['shape_attributes']['all_points_y'][idx_polygon]))
        if len(polygon_calc) < 8: continue
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(list(map(tuple, polygon_calc)), outline=1, fill=255)
        mask = np.array(img)
        mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

        dst = cv2.addWeighted(image, 0.5, mask_stack, 0.2, 0)
        color = (0, 0, 255)
        if i_idx['correct']:
            color = (0, 255, 0)

        factor_area_of_mask = ((max(i_idx['regions']['0']['shape_attributes']['all_points_x']) - min(i_idx['regions']['0']['shape_attributes']['all_points_x'])) *
                                (max(i_idx['regions']['0']['shape_attributes']['all_points_y']) - min(i_idx['regions']['0']['shape_attributes']['all_points_y'])))/(w*h)
        if factor_area_of_mask < 0.0008:
            json_tmp_copy[title]['verified'] = True
            json_tmp_copy[title]['correct'] = False
            continue

        cv2.rectangle(dst, (min(i_idx['regions']['0']['shape_attributes']['all_points_x']),
                            min(i_idx['regions']['0']['shape_attributes']['all_points_y'])),
                      (max(i_idx['regions']['0']['shape_attributes']['all_points_x']),
                       max(i_idx['regions']['0']['shape_attributes']['all_points_y'])), color, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(dst, 'Mask: {}/{} \n Total left: {} '.format(idx_+1, len(i), len(json_tmp_copy) - cnt), (10, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        imS = cv2.resize(dst, (960, 540))  # Resize image
        cv2.imshow('Image', imS)
        res = cv2.waitKey(0)
        print("press button: ", res)
        if res == 32:
            json_tmp_copy[title]['verified'] = True
            json_tmp_copy[title]['correct'] = False
            print("Mask are False")
        elif res == 115:
            print("Wait for save ...")
            with open(path_to_json, 'w') as fp:
                json.dump(json_tmp_copy, fp)
            print("Change saved ! ")
        elif res == 27:
            print("Wait for save ...")
            with open(path_to_json, 'w') as fp:
                json.dump(json_tmp_copy, fp)
            print("Change saved ! ")
            exit(0)
        else:
            json_tmp_copy[title]['verified'] = True
            json_tmp_copy[title]['correct'] = True
            print("Mask are True")
        print("-------------------------")

print("Wait for save ...")
with open(path_to_json, 'w') as fp:
    json.dump(json_tmp_copy, fp)
print("Change saved ! ")