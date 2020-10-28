import json
import numpy as np
from pycocotools import mask
from skimage import measure
import os, json
import cv2
import pandas as pd

fishial_mask_dir = r'createing_dataset/dataset'
list_path = os.listdir(fishial_mask_dir)
result_dict = {}
count = 0

df = pd.read_csv (r'createing_dataset/resources/train-annotations-object-segmentation.csv')
print("Start index: ", len(df.index))
df = df[(df.LabelName == "/m/0ch_cf") | (df.LabelName == "/m/03fj2") | (df.LabelName == "/m/0by6g") | (df.LabelName == "/m/0m53l" ) | (df.LabelName == "/m/0nybt" ) | ( df.LabelName == "/m/0fbdv")]
print("Finish index: ", len(df.index))
df.to_csv('train-annotations-object-segmentation-fish.csv', sep='\t', encoding='utf-8')
df = df.ImageID.value_counts()
print(df)
# for ostalos,  filename in enumerate(list_path):
#     print("Score: ", len(list_path) - ostalos, "Count: ", count)
#     title, ext = os.path.splitext(os.path.basename(filename))
#     ext = ext.lower()
#     if ext == '.png':
#         fullname = os.path.join(fishial_mask_dir, filename)
#         title, ext = os.path.splitext(os.path.basename(fullname))
#         ground_truth_binary_mask = cv2.imread(fullname, 0)
#         ground_truth_binary_mask_zero = np.zeros(ground_truth_binary_mask.shape)
#         fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
#         encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
#         ground_truth_area = mask.area(encoded_ground_truth)
#         ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
#         contours = measure.find_contours(ground_truth_binary_mask, 0.5)
#
#         B = []
#         count += len(contours)
#
#         for contour in contours:
#             contour = np.flip(contour, axis=1)
#             segmentation = contour.ravel().tolist()
#             B = np.reshape(segmentation, (-1, 2))

