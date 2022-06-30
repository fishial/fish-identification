import sys
# Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import os
from module.segmentation_package.interpreter_segm import SegmentationInference
from module.classification_package.src.utils import read_json, save_json
import cv2
import warnings

warnings.filterwarnings('ignore')

from os import listdir
from os.path import isfile, join


def get_only_files(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


input_folder = '../../output/segmentation/amp_on_new_ds/18_06_2022_06_55_38'

list_of_files_in_directory = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
array_of_eval_results = []

for file_name in list_of_files_in_directory:
    splited = os.path.splitext(file_name)
    if splited[1] != '.pth': continue

    print(10 * "*")
    print(f"Model name: {splited[0]}")
    print(10 * "*")

    # Model Init
    model_segmentation = SegmentationInference(os.path.join(input_folder, file_name), device='cuda', class_num=1,
                                               labels_list=['Fish'])
    model_segmentation.re_init_model(0.1)

    d = "../../datasets/no_fish"
    answers_json = {
        'outputs': []
    }
    lisss = [os.path.join(d, o) for o in os.listdir(d)
             if os.path.isdir(os.path.join(d, o))]

    for indices, i in enumerate(lisss):
        onlyfiles = get_only_files(i)
        if not os.path.basename(i).isnumeric():

            for img_url_idx, img_url in enumerate(onlyfiles):
                print(f"[{splited[0]}] Left: {len(onlyfiles) - img_url_idx}/{len(lisss) - indices}", end='\r')
                try:
                    img = cv2.imread(os.path.join(i, img_url))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(e)
                    continue
                meta_data, masks, outputs = model_segmentation.simple_inference(img)
                answers_json['outputs'].extend(meta_data['scores'])

    save_json(answers_json, os.path.join(input_folder, f"{splited[0]}_eval_test_none_fish.json"))
