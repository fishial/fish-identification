import numpy as np
import pandas as pd
import os, json
import xlrd
import os.path as path
import re
from shutil import copyfile

mask_dataset = r'resources/fishial-new/train'
img_dir = r'resources/train-fish-img'
img_dir_test = r'resources/test-fish-img'
new_part = r'resources/new_part'
val = r'resources/val-fish-img'
path_to_unique_img = r'resources/fish-img-unique'
os.makedirs(path_to_unique_img, exist_ok=True)

list_path_img = [os.path.basename(i) for i in os.listdir(mask_dataset)]
list_img_dir = [os.path.basename(i) for i in os.listdir(val)] #+ [os.path.basename(i) for i in os.listdir(img_dir_test)] + [os.path.basename(i) for i in os.listdir(new_part)]  + [os.path.basename(i) for i in os.listdir(val)]
data = set(list_img_dir) - set(list_path_img)

for idx, i in enumerate(data):
    print("Leave: {}".format(len(data) - idx))
    copyfile(os.path.join(val, i), os.path.join(path_to_unique_img, i))