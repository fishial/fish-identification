import imageio
import os, json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw


def change_color_of_mask(path_to_folder, src_color=np.array([32, 11, 119]), dst_color=(255, 255, 255)):
    list_png = os.listdir(path_to_folder)
    for leave, filename in enumerate(list_png):
        print("Score: ", len(list_png) - leave, "count: ")
        title, ext = os.path.splitext(os.path.basename(filename))
        ext = ext.lower()
        if ext == '.png':
            img = cv2.imread(os.path.join(path_to_folder, filename))
            # Mask image to only select color
            mask = cv2.inRange(img, src_color, src_color)
            # Change image
            img[mask > 0] = dst_color
            cv2.imwrite(os.path.join(path_to_folder, filename), img)


def get_format_dict(title, area, array_x, array_y, filename):
    return {
            title: {
                "fileref": "",
                "size": area,
                "filename": filename,
                "base64_img_data": "",
                "verified": False,
                "correct": False,
                "file_attributes": {
                },
                "regions": {
                    "0": {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": array_x,
                            "all_points_y": array_y
                        },
                        "region_attributes": {

                        }
                    }
                }
            }
        }


def remove_background_and_set_color():

    def random_color():
        levels = range(32, 256, 32)
        return tuple(random.choice(levels) for _ in range(3))

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

        for idx_, i_idx in enumerate(i):
            img_test = image.copy()
            color = random_color()
            polygon_calc = []
            for i in range(len(i_idx['regions']['0']['shape_attributes']['all_points_x'])):
                polygon_calc.append((i_idx['regions']['0']['shape_attributes']['all_points_x'][i],
                                     i_idx['regions']['0']['shape_attributes']['all_points_y'][i]))

                # cv2.circle(image, (i_idx['regions']['0']['shape_attributes']['all_points_x'][i],
                #                     i_idx['regions']['0']['shape_attributes']['all_points_y'][i]), 2, color, -1)
            if len(polygon_calc) < 8: continue
            img = Image.new('L', (w, h), 0)
            # polygon_calc = [(0,0), (100,0), (100, 100), (0, 100)]
            ImageDraw.Draw(img).polygon(list(map(tuple, polygon_calc)), outline=1, fill=255)
            mask = np.array(img)
            mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask
            # -- Blend masked img into MASK_COLOR background --------------------------------------
            mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
            img_test = img_test.astype('float32') / 255.0  # for easy blending
            masked = (mask_stack * img_test) + ((1 - mask_stack) * (0.0, 0.0, 1.0))  # Blend
            masked = (masked * 255).astype('uint8')  # Convert back to 8-bit
            # dst = cv2.addWeighted(image, 0.5, mask_stack, 0.5, 0)
            cv2.imshow('img', masked)  # Display
            cv2.waitKey()
