import cv2
import numpy as np
import logging
import torch
import copy
import time
import torchvision.models as models

from torchvision import transforms
from torch import nn
from PIL import Image
from module.classification_package.src.model import EmbeddingModel, Backbone
from module.classification_package.src.utils import read_json


class EmbeddingClassifier:
    def __init__(self, model_path, data_set, device='cpu'):
        self.device = device
        start_time = time.time()
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

        self.model_path = model_path
        
        resnet18 = models.resnet18()
        resnet18.fc = nn.Identity()

        backbone = Backbone(resnet18)
        self.model = EmbeddingModel(backbone)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        self.model.to(device)
        
        self.loader = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if type(data_set) != str:
            self.data_base = self.__get_data_base(data_set)
        else:
            self.data_base = read_json(data_set)
        logging.info("Initialization finished in {} [s]".format(round(time.time() - start_time, 2)))

    def simple_inference(self, img, top_k = 15):
        image = Image.fromarray(img)
        image = self.loader(image).float()
        image = torch.tensor(image)
        
        dump_embed = self.model(image.unsqueeze(0)).detach().numpy()
        topest = self.__classify(dump_embed)
        
        flatten = [iii[0] for iii in topest[:top_k]]
        my_dict = [[i, flatten.count(i)] for i in flatten]
        my_dict = self.__remove_dupliceta(my_dict)
        my_dict = sorted(my_dict, key=lambda x: x[1], reverse=True)
        return [[match[0], match[1]/top_k] for match in my_dict]

    def __classify(self, embedding):
        classification_lib = []

        for k in self.data_base:
            for kk in self.data_base[k]['vectors']:
                distance = np.abs(embedding - np.array(kk)).sum()
                classification_lib.append([k, distance])
        classification_lib = sorted(classification_lib, key=lambda x: x[1], reverse=False)
        return classification_lib

    def __get_data_base(self, data_set):
        data_with_vectors = copy.deepcopy(data_set.library_name)
        for i in range(len(data_set)):
            logging.info("Left: {}".format(len(data_set) - i))
            output = self.model(data_set[i][0].unsqueeze(0).to(self.device)).detach().numpy()
            converted_list = output[0]
            if "vectors" in data_with_vectors[int(data_set[i][1])]:
                data_with_vectors[int(data_set[i][1])]["vectors"].append(converted_list)
            else:
                data_with_vectors[int(data_set[i][1])].update({"vectors": [converted_list]})
        return data_with_vectors
    
    @staticmethod
    def __remove_dupliceta(mylist):
        seen = set()
        newlist = []
        for item in mylist:
            t = tuple(item)
            if t not in seen:
                newlist.append(item)
                seen.add(t)
        return newlist
