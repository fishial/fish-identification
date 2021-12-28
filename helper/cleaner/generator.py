#!/usr/bin/env python3
"""COCO Dataset Viewer.

View images with bboxes from the COCO dataset.
"""
import sys
sys.path.insert(1, '/Users/admin/Desktop/Codahead/Object-Detection-Model')

import argparse
import os
import json
import logging
import urllib.request as ur
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

mask_folder = "../../data/tmp_mask"
onlyfiles = [f for f in listdir(mask_folder) if isfile(join(mask_folder, f))]
json_file = {}
test = []
for i in onlyfiles:
    info = i.replace(".png", "").split("_")
    test.append(info[1])
    if info[0] not in json_file:
        json_file.update({
            info[0]: [info[1]]
        })
    else:
        json_file[info[0]].append(info[1])

print(len(json_file))
print(json_file)
with open("../../data/invalid_id.json", 'w', encoding='utf-8') as f:
    json.dump(json_file, f)