import numpy as np
import pandas as pd
import os, json
import xlrd
import os.path as path
import re
from shutil import copyfile

# Settings
# Set ipython's max row display
pd.set_option('display.max_row', 1000)
# Set iPython's max column width to 50
pd.set_option('display.max_columns', 1000)

# directory with mask
name_root_fir = 'val'
path_to_csv_file = r'resources/validation-annotations-object-segmentation-fish.csv'
fishial_mask_dir = os.path.join(r'resources', name_root_fir)
fishial_mask_dir_array = []


def find_row_on_df(data, path):
    idx = data['MaskPath'].loc[lambda x: x==path].index
    if len(idx) > 1:
        print("COUNT: ", len(idx))
    if idx is not None and len(idx) > 0:
        return data['LabelName'][idx[0]]
    return None

# dictionary with choosed classes
dict = json.load(open(r'resources/category.json'))
array_code_and_path = []
df = pd.read_csv(path_to_csv_file)

main_path = os.path.join(r"resources", name_root_fir+"-fish")
os.makedirs(main_path, exist_ok=True)
for i in dict['Category']:
    array_code_and_path.append([i['code'], main_path])
    for z in i['Subcategory']:
        for x in z:
            array_code_and_path.append([z[x], main_path])
list_path = os.listdir(fishial_mask_dir)

for ostalos,  filename in enumerate(list_path):
    print("Score: ", len(list_path) - ostalos, "count: ", len(fishial_mask_dir_array))
    title, ext = os.path.splitext(os.path.basename(filename))
    ext = ext.lower()
    if ext == '.png':
        fullname = os.path.join(fishial_mask_dir, filename)
        title, ext = os.path.splitext(os.path.basename(fullname))
        answear = find_row_on_df(df, filename)
        if answear is None: continue
        for code_i in array_code_and_path:
            if code_i[0] == answear:
                src_path = os.path.join(fishial_mask_dir, filename)
                dst_path =  os.path.join(code_i[1], filename)
                fishial_mask_dir_array.append(filename)
                copyfile(src_path, dst_path)
                break

print(len(fishial_mask_dir_array))