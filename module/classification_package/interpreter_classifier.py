import cv2
import numpy as np
import logging
import torch
import time
import torchvision.models as models

from PIL import Image
from torchvision import transforms
from torch import nn


class Backbone(nn.Module):
    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor):
        return self.resnet(x)


class FcNet(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes):
        super().__init__()
        self.backbone = backbone
        self.fc_1 = nn.Linear(512, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.fc_1(x)
        return x


class ClassifierFC:

    def __init__(self, model_path, n_classes=61, device='cpu'):
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        start_time = time.time()
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

        self.model_path = model_path

        resnet18 = models.resnet18()
        resnet18.fc = nn.Identity()

        backbone = Backbone(resnet18)

        self.model = FcNet(backbone, n_classes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(device)))
        self.model.eval()
        self.model.to(device)

        self.loader = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        logging.info("Initialization ClassifierFC finished in {} [s]".format(round(time.time() - start_time, 2)))

    def inference(self, image, top_k=6):
        start_time = time.time()
        dump = self.softmax(self.model(image.unsqueeze(0)))
        output = torch.topk(dump, top_k)
        #         logging.info(
        #             "Inference time by classification model has taken {} [s]".format(round(time.time() - start_time, 2)))
        return [[int(output.indices[0][match]), round(float(output.values[0][match]), 5)] for match in
                range(len(output.indices[0]))]

    def inference_numpy(self, img, top_k=6):
        image = Image.fromarray(img)
        image = self.loader(image).float()
        image = torch.tensor(image)
        return self.inference(image, top_k)

    def batch_inference(self, imgs, top_k=6):
        start_time = time.time()
        batch_input = []
        for idx in range(len(imgs)):  # assuming batch_size=len(imgs)
            image = Image.fromarray(imgs[idx])
            image = self.loader(image).float()
            image = torch.tensor(image)
            batch_input.append(image)

        batch_input = torch.stack(batch_input)
        dump = self.softmax(self.model(batch_input))
        output = torch.topk(dump, top_k)
        logging.info(
            "Inference time by classification model has taken {} [s]".format(round(time.time() - start_time, 2)))
        return [
            [[int(output.indices[output_idx][match]), round(float(output.values[output_idx][match]), 3)] for match in
             range(len(output.indices[output_idx]))] for output_idx in range(len(output.indices))]