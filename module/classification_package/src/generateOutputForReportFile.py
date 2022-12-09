import numpy as np
import logging
import torch
import random
import json
import torchvision.models as models
import os
import sys

# Change path specificly to your directories
sys.path.insert(1, '/home/fishial/Fishial/Object-Detection-Model')

from module.classification_package.src.utils import read_json, save_json
from torchvision import transforms
from torch import nn
from PIL import Image


import time
import torch
import os
import cv2
import math 
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.neighbors import KDTree
import pandas as pd
from module.classification_package.src.utils import read_json, save_json
from module.classification_package.interpreter_classifier import ClassifierFC
from module.classification_package.interpreter_embeding import EmbeddingClassifier
# from module.classification_package.interpreter_embeding_data import EmbeddingClassifierData

from module.segmentation_package.interpreter_segm import SegmentationInference
from module.classification_package.src.dataset import FishialDataset, FishialDatasetOnlineCuting
# from module.segmentation_package.src.utils import resize_image
from PIL import Image
import numpy as np
import random
import sklearn.metrics.pairwise
import scipy.spatial.distance
import copy
import json
import time
import requests
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from os import listdir
from os.path import isfile, join
# TRESHOLD 40.6256
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (40, 20),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def classify(data_base, embedding):
        diff = (data_base - embedding).pow(2).sum(dim=2).sqrt()
        val, indi = torch.sort(diff)
        class_lib = []
        for idx, i in enumerate(val):
            for dist_id, dist in enumerate(i[:25]):
                if dist == 0.0: continue
                if data_base[idx][indi[idx][dist_id]].sum() > 10000: continue
                class_lib.append([idx, dist])
        class_lib = sorted(class_lib, key=lambda x: x[1], reverse=False)
        return class_lib[:1][0]
    
folder_path = r'/home/fishial/Fishial/output/classification/resnet_18_184_train_06_12'
data_sets = {'full': {} }

embedding = torch.load(os.path.join(folder_path, 'embeddings.pt')).to('cpu')
data_labels = read_json(os.path.join(folder_path, 'labels.json'))
data_labels_list = [data_labels[zxc] for zxc in data_labels]
data_indices = read_json(os.path.join(folder_path, 'idx.json'))

dataset_name = 'full'
data_sets[dataset_name].update({
    'labels': {
        data_labels[label]: {       
        'pred':     [],
        'distance': [],
        'idx': []
} for label in data_labels}})

for label_id in range(embedding.shape[0]):
    for idx_ann in range(len(data_indices[str(label_id)]['annotation_id'])):
        
        idx_ann_s = data_indices[str(label_id)]['annotation_id'][idx_ann]
        label_correct = data_labels[str(label_id)]

        output = classify(embedding, embedding[label_id][idx_ann])
        if label_correct in data_labels_list:
            print(label_correct, data_labels[str(output[0])])
            data_sets[dataset_name]['labels'][label_correct]['distance'].append(output[1].item())
            data_sets[dataset_name]['labels'][label_correct]['pred'].append(data_labels[str(output[0])])
            data_sets[dataset_name]['labels'][label_correct]['idx'].append(idx_ann_s)

        print("name: {} Left: [{}/{}/{}] ".format(dataset_name, 
                                                     idx_ann, label_id, embedding.shape[0]), end='\r')
            
save_json(data_sets,os.path.join(folder_path, "output_test_base.json"))