import imgaug as ia
import os, json
from shutil import copyfile
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon
from createing_dataset.src.helper import get_format_dict
import warnings


def save_dict(dictionary, path):
    with open(path, 'w') as fp:
        json.dump(dictionary, fp)


def convert_aug_pol_to_two_array(poi, image):
    array_of_polygon = []
    for i in poi:
        x_array = []
        y_array = []
        for idx, z in enumerate(i):
            x = int(z[0] if int(z[0]) > 0 else 0)
            y = int(z[1] if int(z[1]) > 0 else 0)
            if (x and y) == 0: continue
            x_array.append(x)
            y_array.append(y)
        array_of_polygon.append([x_array, y_array])
    return array_of_polygon

# import data
path_to_json = r'fishial/train/via_region_data.json'
# path to augmented dataset
path_to_aug_dataset = r'fishial/train-aug'
# count of augmented img
cnt_aug = 2
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

aug1 = iaa.Sequential([
            iaa.Fliplr(0.3),
            iaa.Flipud(0.6),
            iaa.Add((-40, 40))])
aug2 = iaa.JpegCompression(compression=(70, 99))
aug3 = iaa.Affine(rotate=(-45, 45))
aug4 = iaa.AdditiveGaussianNoise(scale=0.08*255, per_channel=True)
# create folder for augumented dataset !
os.makedirs(path_to_aug_dataset, exist_ok=True)
result_dict = {}

for leave, i in enumerate(unique_img_array):
    print("Score: ", len(unique_img_array) - leave, len(result_dict))
    img_main = os.path.join(r'fishial/train', i[0]['filename'])
    dst_path = os.path.join(path_to_aug_dataset, os.path.basename(i[0]['filename']))
    copyfile(img_main, dst_path)
    image = cv2.imread(img_main)
    array_of_polygon = []
    for idx_, i_idx in enumerate(i):
        result_dict.update(get_format_dict(
            i_idx['filename'] + "_main_" + str(idx_),
            i_idx['size'],
            i_idx['regions']['0']['shape_attributes']['all_points_x'],
            i_idx['regions']['0']['shape_attributes']['all_points_y'],
            i_idx['filename']))

        polygon_tmp = []
        for i in range(len(i_idx['regions']['0']['shape_attributes']['all_points_x'])):
            polygon_tmp.append((i_idx['regions']['0']['shape_attributes']['all_points_x'][i],
                                i_idx['regions']['0']['shape_attributes']['all_points_y'][i]))
        array_of_polygon.append(Polygon(polygon_tmp))
    psoi = ia.PolygonsOnImage(array_of_polygon, shape=image.shape)

    for cnt_aug_idx in range(cnt_aug):

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("error")
                if cnt_aug_idx == 0:
                    image_aug, psoi_aug = aug1(image=image, polygons=psoi)
                elif cnt_aug_idx == 1:
                    image_aug, psoi_aug = aug2(image=image, polygons=psoi)
                elif cnt_aug_idx == 2:
                    image_aug, psoi_aug = aug3(image=image, polygons=psoi)
                else:
                    image_aug, psoi_aug = aug4(image=image, polygons=psoi)
        except Exception:
            print("Error ! name file: {} ".format(img_main))
            continue

        array_of_polygon = convert_aug_pol_to_two_array(psoi_aug, image_aug)
        # save aug image
        title, ext = os.path.splitext(os.path.basename(img_main))
        aug_image_name = '{}_aug_{}{}'.format(title, cnt_aug_idx, ext)
        cv2.imwrite(os.path.join(path_to_aug_dataset, aug_image_name), image_aug)
        width, h, _ = image_aug.shape
        for idx_single_polygon, single_converted_poly in enumerate(array_of_polygon):
            if len(single_converted_poly[0]) < 20: continue
            result_dict.update(get_format_dict(
                aug_image_name + "_{}".format(idx_single_polygon),
                width * h,
                single_converted_poly[0],
                single_converted_poly[1],
                aug_image_name))
save_dict(result_dict, os.path.join(path_to_aug_dataset, 'via_region_data.json'))
