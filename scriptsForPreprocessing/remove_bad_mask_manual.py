import os, json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw


def resize_image_if_big(image):
    h, w, _ = image.shape
    scale_X = 1
    scale_Y = 1
    if w > 1024 or h > 768:
        scale_X = 1024 / w
        scale_Y = 768 / h
        image = cv2.resize(dst, (int(w * scale_X), int(h * scale_Y)))  # Resize image
    cv2.imshow("img", image)
    return scale_X, scale_Y


def click_and_crop(event, x, y, flags, param):
    global points, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        points.append([x, y])


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels) for _ in range(3))


def calc_area(axis):
    return (axis[2] - axis[0]) * (axis[3] - axis[1])


def bubble_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if calc_area(arr[j][0]) > calc_area(arr[j + 1][0]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def draw_mask_on_image(img_src, i, target_mask):
    h, w, _ = img_src.shape
    dst = img_src.copy()
    mask_general = np.zeros((h, w))
    current_idx_on_image = []
    cnt = 0
    for idx_, i_idx in enumerate(i):
        if i_idx[1] in target_mask: continue

        color = random_color()
        polygon_calc = []
        for polygon_idx in range(len(i_idx[0]['regions']['0']['shape_attributes']['all_points_x'])):
            polygon_calc.append((i_idx[0]['regions']['0']['shape_attributes']['all_points_x'][polygon_idx],
                                 i_idx[0]['regions']['0']['shape_attributes']['all_points_y'][polygon_idx]))
            if polygon_idx == 0: continue
            cv2.line(dst, (i_idx[0]['regions']['0']['shape_attributes']['all_points_x'][polygon_idx - 1],
                           i_idx[0]['regions']['0']['shape_attributes']['all_points_y'][polygon_idx - 1]),
                     (i_idx[0]['regions']['0']['shape_attributes']['all_points_x'][polygon_idx],
                      i_idx[0]['regions']['0']['shape_attributes']['all_points_y'][polygon_idx]), color, thickness=2)

        if len(polygon_calc) < 8:
            json_tmp_copy[i_idx[1]]['verified'] = True
            json_tmp_copy[i_idx[1]]['correct'] = False
            continue
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(list(map(tuple, polygon_calc)), outline=1, fill=255)
        tmp_mask = np.array(img)
        mask_general = cv2.addWeighted(tmp_mask, 1, mask_general, 1, 0, dtype=cv2.CV_8UC1)
        current_idx_on_image.append([[min(i_idx[0]['regions']['0']['shape_attributes']['all_points_x']),
                                      min(i_idx[0]['regions']['0']['shape_attributes']['all_points_y']),
                                      max(i_idx[0]['regions']['0']['shape_attributes']['all_points_x']),
                                      max(i_idx[0]['regions']['0']['shape_attributes']['all_points_y'])], i_idx[1]])
        cv2.rectangle(dst, (min(i_idx[0]['regions']['0']['shape_attributes']['all_points_x']),
                            min(i_idx[0]['regions']['0']['shape_attributes']['all_points_y'])),
                      (max(i_idx[0]['regions']['0']['shape_attributes']['all_points_x']),
                       max(i_idx[0]['regions']['0']['shape_attributes']['all_points_y'])), color, 3)
        cnt +=1


    mask_stack = np.dstack([mask_general] * 3)
    mask_stack_arr = np.asarray(mask_stack)
    dst = cv2.addWeighted(dst, 0.5, mask_stack_arr, 0.5, 0, dtype=cv2.CV_8UC3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dst, 'Mask: {} '.format(cnt), (10, 50),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if len(current_idx_on_image) != 1: bubble_sort(current_idx_on_image)

    return dst, current_idx_on_image


points = []
cropping = False

# import data
path_to_json = r'resources/fishial/val/via_region_data.json'
# path to augmented dataset
path_to_aug_dataset = r'resources/fishial/val'
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

result_dict = {}
ok = 0
nok = 0
for leave, current_collection in enumerate(unique_img_array):
    print("Score: ", len(unique_img_array) - leave, len(result_dict))

    # if current_collection[0][0]['verified']:
    #     continue
    img_main = os.path.join(path_to_aug_dataset, current_collection[0][0]['filename'])
    image = cv2.imread(img_main)
    idx_to_remove = []
    dst, current_idx_on_image = draw_mask_on_image(image, current_collection, idx_to_remove)

    original_mask_on_image = current_idx_on_image.copy()
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", click_and_crop)
    points = []
    scale_X, scale_Y = resize_image_if_big(dst)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print("Wait for save ...")
            with open(path_to_json, 'w') as fp:
                json.dump(json_tmp_copy, fp)
            print("Change saved ! ")
        elif key == ord("x"):
            print("Undo")
            try:
                del idx_to_remove[-1]
                dst, current_idx_on_image = draw_mask_on_image(image, current_collection, idx_to_remove)
                scale_X, scale_Y = resize_image_if_big(dst)
            except:
                print("error")
        elif key == 32:
            print("Next image")
            for zx in original_mask_on_image:
                idx_record = zx[1]
                if idx_record in idx_to_remove:
                    json_tmp_copy[idx_record]['verified'] = True
                    json_tmp_copy[idx_record]['correct'] = False
                else:
                    json_tmp_copy[idx_record]['verified'] = True
                    json_tmp_copy[idx_record]['correct'] = True
            break
        elif key == 27:
            print("Wait for save ...")
            with open(path_to_json, 'w') as fp:
                json.dump(json_tmp_copy, fp)
            print("Change saved ! ")
            exit(0)
        if len(points) == 0: continue
        idx_to_remove_tmp = []
        for point in points:
            for i in current_idx_on_image:
                if i[0][0] < point[0] / scale_X < i[0][2] and i[0][1] < point[1]/scale_Y < i[0][3]:
                    idx_to_remove_tmp.append(i[1])
                    break
        if len(idx_to_remove_tmp) != 0:
            idx_to_remove += idx_to_remove_tmp
            dst, current_idx_on_image = draw_mask_on_image(image, current_collection, idx_to_remove)
            scale_X, scale_Y = resize_image_if_big(dst)
        points = []
print("Wait for save ...")
with open(path_to_json, 'w') as fp:
    json.dump(json_tmp_copy, fp)
print("Change saved ! ")