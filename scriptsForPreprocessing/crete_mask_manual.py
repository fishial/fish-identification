import os, json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


points = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    global points, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))


# import data
path_to_json = r'resources/fishial/train/via_region_data.json'

# path to augmented dataset
path_to_aug_dataset = r'resources/fishial/train'
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
    dst = image.copy()
    mask_general = np.zeros((h, w))
    print("Genreal: ", mask_general.shape)
    for idx_, i_idx in enumerate(i):
        color = random_color()
        polygon_calc = []
        for polygon_idx in range(len(i_idx['regions']['0']['shape_attributes']['all_points_x'])):
            polygon_calc.append((i_idx['regions']['0']['shape_attributes']['all_points_x'][polygon_idx],
                                 i_idx['regions']['0']['shape_attributes']['all_points_y'][polygon_idx]))
        if len(polygon_calc) < 8: continue
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(list(map(tuple, polygon_calc)), outline=1, fill=255)
        tmp_mask = np.array(img)
        mask_general = cv2.addWeighted(tmp_mask, 1, mask_general, 1, 0, dtype=cv2.CV_8UC1)

        cv2.rectangle(dst, (min(i_idx['regions']['0']['shape_attributes']['all_points_x']),
                            min(i_idx['regions']['0']['shape_attributes']['all_points_y'])),
                      (max(i_idx['regions']['0']['shape_attributes']['all_points_x']),
                       max(i_idx['regions']['0']['shape_attributes']['all_points_y'])), color, 3)
    mask_stack = np.dstack([mask_general] * 3)
    mask_stack_arr = np.asarray(mask_stack)
    dst = cv2.addWeighted(dst, 0.5, mask_stack_arr, 0.5, 0, dtype=cv2.CV_8UC3)
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", click_and_crop)

    points = []
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("img", dst)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            print("You press r")
        elif key == 32:
            break
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        if cropping:
            if len(points)==1:
                cv2.circle(dst, (int(points[0][0]), int(points[0][1])), 2, (0, 255, 0), -1)
            elif len(points) > 1:
                cv2.circle(dst, (int(points[len(points) - 1][0]),
                                 int(points[len(points) - 1][1])), 2, (0, 255, 0), -1)
                cv2.line(dst, (int(points[len(points) - 2][0]),
                                 int(points[len(points) - 2][1])),
                         (int(points[len(points) - 1][0]),
                          int(points[len(points) - 1][1])), (0, 255, 0), thickness=2)

            cv2.imshow("img", dst)
            cropping = False

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(points) > 3:
        print("Polygon: {} point: {}".format(len(points), points))

