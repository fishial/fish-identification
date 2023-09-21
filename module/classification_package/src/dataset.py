import sys

# Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import pandas as pd
import collections
import numpy as np
import os.path
import torch
import os

import random
import pyclipper
import time
import copy
import math
import cv2

from PIL import Image
from os import walk
from shapely.geometry import Polygon, Point
from shapely import geometry
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler
from module.segmentation_package.src.utils import get_mask
from module.classification_package.src.utils import read_json


class FishialDatasetFoOnlineCuting(Dataset):
    def __init__(self,
                 records,
                 labels_dict,
                 train_state=False,
                 transform=None,
                 crop_type = 'poly'):
        
        #Add internal id by dictionary
        for label in records:
            for k in records[label]:
                k.update({'id_internal': labels_dict[label]})
                
        self.labels_dict = labels_dict
        self.train_state = train_state
        self.crop_type = crop_type
        self.transform = transform
        
        self.data_compleated = []
        for label in records:
            self.data_compleated.extend(records[label])
    
        self.n_classes = len(set([i['name'] for i in self.data_compleated]))
        self.targets = [i['id_internal'] for i in self.data_compleated]
        
        
    def __get_margin(self, poly):
        # create example polygon
        poly = geometry.Polygon(poly)

        # get minimum bounding box around polygon
        box = poly.minimum_rotated_rectangle

        # get coordinates of polygon vertices
        x, y = box.exterior.coords.xy

        # get length of bounding box edges
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

        # get length of polygon as the longest edge of the bounding box
        length = max(edge_length)

        # get width of polygon as the shortest edge of the bounding box
        width = min(edge_length)
        marg = int(min(width, length) * 0.04)
        random_margin = random.randint(-marg, int(marg * 1.9))
        return random_margin

    def __shrink_poly(self, poly, value, img_shape):
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(value)

        for i in range(len(solution[0])):
            solution[0][i][0] = max(0, min(img_shape[1], solution[0][i][0]))
            solution[0][i][1] = max(0, min(img_shape[0], solution[0][i][1]))
        return solution
    
    def __get_poly_mask(self, img_path, polyline_main):
        
        image = cv2.imread(img_path)
        if self.train_state:
            margine = self.__get_margin(polyline_main)
            try:
                polyline_main = self.__shrink_poly(polyline_main, margine, image.shape[:2])[0]
            except:
                print(f"Error: {img_path}")
                pass
        mask = get_mask(image, np.array(polyline_main))
        
        return mask
    
    def __get_cropped_rect(self, img_path):

        img_path = self.data_compleated[idx]['file_name']
        image = cv2.imread(img_path)
        return image
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.data_compleated)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        img_path = self.data_compleated[idx]['file_name']
        polyline_main = self.data_compleated[idx]['poly']
        
        if self.crop_type == 'poly':
            image = self.__get_poly_mask(img_path, polyline_main)
        else:
            image = self.__get_cropped_rect(img_path)
            
        mask = Image.fromarray(image) 
        if self.transform:
            mask = self.transform(mask)
        
        return (mask, torch.tensor(self.data_compleated[idx]['id_internal']))
    

class BalancedBatchSampler(BatchSampler):
    """
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader: # image_tensor, label, dict_description
            self.labels_list.append(label)
            
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size