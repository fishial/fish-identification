import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import os
import cv2
import json
import copy
import warnings
import imgaug as ia
import imgaug.augmenters as iaa

from shutil import copyfile
from imgaug.augmentables.polys import Polygon
from tqdm import tqdm

from module.classification_package.src.utils import save_json


# The script will create an augumentation images, according the loader(transformer).
# That script we use if need us to use 'offline' data augumentation.

def get_max_id(data):
    ids = []

    for i in data['images']:
        if 'id' in i:
            ids.append(i['id'])
    return max(ids)


def convert_aug_pol_to_two_array(poi):
    array_of_polygon = []
    for i in poi:
        array_segm = []
        for idx, z in enumerate(i):
            x = int(z[0] if int(z[0]) > 0 else 0)
            y = int(z[1] if int(z[1]) > 0 else 0)
            if (x and y) == 0: continue
            array_segm.append(x)
            array_segm.append(y)
        array_of_polygon.append(array_segm)
    return array_of_polygon


# import data
path_to_json = r'fishial_collection/fishial_collection_correct.json'
# path to augmented dataset
path_to_aug_dataset = r'fishial_collection/Train-aug'
img_path_main = r'fishial_collection/Train'
# count of augmented img
cnt_aug = 5

aug2 = iaa.JpegCompression(compression=(70, 99))
aug3 = iaa.Affine(rotate=(-45, 45))
aug4 = iaa.AdditiveGaussianNoise(scale=0.08 * 255, per_channel=True)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                       # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               second=iaa.LinearContrast((0.5, 2.0))
                           )
                       ]),
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

json_tmp = json.load(open(path_to_json))

unique_img_array = []

os.makedirs(path_to_aug_dataset, exist_ok=True)

result_dict = copy.deepcopy(json_tmp)

bodyes_shapes_ids = []
for i in json_tmp['categories']:
    if i['name'] == 'General body shape':
        bodyes_shapes_ids.append(int(i['id']))

for img_ma in tqdm(range(len(json_tmp['images']))):
    if 'train_data' in json_tmp['images'][img_ma]:
        if not json_tmp['images'][img_ma]['train_data']: continue
        img_main = os.path.join(img_path_main, json_tmp['images'][img_ma]['file_name'])
        dst_path = os.path.join(path_to_aug_dataset, os.path.basename(json_tmp['images'][img_ma]['file_name']))
        copyfile(img_main, dst_path)
        image = cv2.imread(dst_path)
        single_img = []

        for ann in json_tmp['annotations']:
            if 'segmentation' in ann and ann['image_id'] == json_tmp['images'][img_ma]['id'] and ann[
                'category_id'] in bodyes_shapes_ids:
                single_img.append(Polygon(ann['segmentation']))

        psoi = ia.PolygonsOnImage(single_img, shape=image.shape)

        for cnt_aug_idx in range(cnt_aug):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("error")
                    if cnt_aug_idx == 0:
                        image_aug, psoi_aug = seq(image=image, polygons=psoi)
                    elif cnt_aug_idx == 1:
                        image_aug, psoi_aug = aug2(image=image, polygons=psoi)
                    elif cnt_aug_idx == 2:
                        image_aug, psoi_aug = aug3(image=image, polygons=psoi)
                    else:
                        image_aug, psoi_aug = seq(image=image, polygons=psoi)

                array_of_polygon = convert_aug_pol_to_two_array(psoi_aug)
                # save aug image
                title, ext = os.path.splitext(os.path.basename(img_main))
                if ext == '':
                    ext = '.png'
                aug_image_name = '{}_aug_{}{}'.format(title, cnt_aug_idx, ext)
                cv2.imwrite(os.path.join(path_to_aug_dataset, aug_image_name), image_aug)

                # save new image record to json
                tmp_img_dict = json_tmp['images'][img_ma]
                tmp_img_dict['id'] = get_max_id(result_dict)
                tmp_img_dict['file_name'] = aug_image_name
                result_dict['images'].append(tmp_img_dict)

                for idx_single_polygon, single_converted_poly in enumerate(array_of_polygon):
                    if len(single_converted_poly) < 10: continue
                    tmp_poly = {
                        'segmentation': single_converted_poly,
                        'image_id': tmp_img_dict['id'],
                        'category_id': 1
                    }
                    result_dict['annotations'].append(tmp_poly)
            except Exception:
                print("Error ! name file: {} ".format(img_main))
                continue
save_json(result_dict, os.path.join('fishial_collection', 'fishial_collection_correct_aug.json'))