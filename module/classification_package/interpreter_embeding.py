import numpy as np
import logging
import torch
import random
import json
import torchvision.models as models

from torchvision import transforms
from torch import nn
from PIL import Image


class EmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, last_layer=512, emb_dim=128):
        super().__init__()
        self.backbone = backbone
        self.embeddings = nn.Linear(last_layer, emb_dim)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        return self.embeddings(self.backbone(x))


# The classifier works as follows.
# Initialization
# 1. We load the weights into the trained neural network.
# 2. We load the dataset vector into the buffer into the pie torch tensor.
# 3. We process weights by removing records that have large intra-class outliers.

# Inference
# 1. Getting a vector
# 2. Compilation of a list H of the number of closest distances.
# 3. Compilation of a list of the most frequent classes.
# 4. Converting distances to probability using a linear relationship in a given range.
# 5. Returning a List of Classes

class EmbeddingClassifier:
    def __init__(self, model_path, data_set_path, data_id_path, device='cpu', THRESHOLD=4.76, resnet=models.resnet18()):
        self.device = device
        self.THRESHOLD = THRESHOLD
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

        self.model_path = model_path
        self.indexes_of_elements = self.read_json(data_id_path)

        resnet18 = resnet
        resnet18.fc = nn.Identity()

        self.model = EmbeddingModel(resnet, 512, 256)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(device)))
        self.model.eval()
        self.model.to(device)

        self.loader = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.data_base = self.__clear_data(torch.load(data_set_path).to(device), 256)

    def inference(self, image, top_k=6):
        dump_embed = self.model(image.unsqueeze(0).to(self.device)).detach()

        topest, to_hell = self.__classify(dump_embed[0])
        if len(topest) > 0:

            my_dict = [[i[0], [iii[0] for iii in topest[:top_k]].count(i[0]), i[1], i[2]] for i in topest[:top_k]]
            my_dict = sorted(my_dict, key=lambda x: x[1], reverse=True)
            my_dict = [[match[0], match[2], match[3]] for match in my_dict]
            my_dict = self.__get_confs(my_dict)
            my_dict = self.remove_duplicate(my_dict)
            my_dict = my_dict[:1] + sorted(my_dict[1:], key=lambda x: x[1], reverse=False)
            return my_dict
        else:
            my_dict = [[i[0], [iii[0] for iii in to_hell[:top_k]].count(i[0]), i[1], i[2]] for i in to_hell[:top_k]]
            my_dict = sorted(my_dict, key=lambda x: x[1], reverse=True)
            my_dict = [[match[0], match[2], match[3]] for match in my_dict]
            return [[my_dict[0][0], 0.01]]

    def inference_numpy(self, img, top_k=6):
        image = Image.fromarray(img)
        image = self.loader(image).float()
        image = torch.tensor(image)
        return self.inference(image, top_k)

    def __get_confidence(self, dist):
        min_dist = 2.62
        max_dist = self.THRESHOLD
        delta = max_dist - min_dist
        return 1.0 - (max(min(max_dist, dist), min_dist) - min_dist) / delta

    def __get_confs(self, my_dict):

        list_of_dist = [i[1] for i in my_dict]
        list_old_conf = [self.__get_confidence(i) for i in list_of_dist]

        delta = (max(list_of_dist) - min(list_of_dist))
        local_conf = [(i - min(list_of_dist)) / delta for i in list_of_dist]

        final_conf = [max(0.09, (float(x) - float(y))) for x, y in zip(list_old_conf, local_conf)]
        my_dict = [[i[0], round(final_conf[idx], 3), i[2]] for idx, i in enumerate(my_dict)]

        return my_dict

    def batch_inference(self, imgs, top_k=8):
        batch_input = []
        for idx in range(len(imgs)):  # assuming batch_size=len(imgs)
            image = Image.fromarray(imgs[idx])
            image = self.loader(image).float()
            image = torch.tensor(image)
            batch_input.append(image)

        batch_input = torch.stack(batch_input)
        dump_embed = self.model(batch_input).detach()

        outputs = []
        for dump in dump_embed:
            topest, to_hell = self.__classify(dump)

            if len(topest) > 0:

                my_dict = [[i[0], [iii[0] for iii in topest[:top_k]].count(i[0]), i[1], i[2]] for i in topest[:top_k]]
                my_dict = sorted(my_dict, key=lambda x: x[1], reverse=True)
                my_dict = [[match[0], match[2], match[3]] for match in my_dict]
                my_dict = self.__get_confs(my_dict)
                my_dict = self.remove_duplicate(my_dict)
                my_dict = my_dict[:1] + sorted(my_dict[1:], key=lambda x: x[1], reverse=True)

                outputs.append(my_dict)
            else:

                my_dict = [[i[0], [iii[0] for iii in to_hell[:top_k]].count(i[0]), i[1], i[2]] for i in to_hell[:top_k]]

                my_dict = sorted(my_dict, key=lambda x: x[1], reverse=True)

                my_dict = [[match[0], match[2], match[3]] for match in my_dict]

                outputs.append([[my_dict[0][0], 0.01, my_dict[0][2]]])

        return outputs

    def __classify(self, embedding):
        diff = (self.data_base - embedding).pow(2).sum(dim=2).sqrt()
        val, indi = torch.sort(diff)
        class_lib = []
        to_hell = []
        for idx, i in enumerate(val):
            # return only top N(12) elements from each class
            for dist_id, dist in enumerate(i[:25]):
                to_hell.append([idx, dist])
                if dist == 0.0: continue
                if dist < self.THRESHOLD:
                    if self.data_base[idx][indi[idx][dist_id]].sum() > 10000: continue

                    ann_iddds = self.indexes_of_elements[str(idx)][indi[idx][dist_id]]
                    class_lib.append([idx, dist, ann_iddds])
        class_lib = sorted(class_lib, key=lambda x: x[1], reverse=False)
        to_hell = sorted(to_hell, key=lambda x: x[1], reverse=False)
        return class_lib, to_hell

    def __clear_data(self, data_base, emb_vec):
        list_numbers = random.choices([100, 100], k=emb_vec)
        random_numbers = torch.Tensor(list_numbers)

        for class_id, i in enumerate(data_base):
            tmp_distances = []
            for emb_a_id, emb_a in enumerate(i):
                if emb_a.sum() > 10000: continue
                dist_per_img = []
                for emb_b_id, emb_b in enumerate(i):
                    if emb_b.sum() > 10000: continue
                    if emb_b_id == emb_a_id: continue
                    diff = (emb_a - emb_b).pow(2).sum().sqrt()
                    dist_per_img.append(float(diff))

                tmp_distances.append(np.mean(dist_per_img))
            mean_per_class = np.mean(tmp_distances)
            for k_id, k in enumerate(tmp_distances):
                if k / mean_per_class > 1.4 and len(tmp_distances) > 35:
                    data_base[class_id][k_id] = random_numbers
        return data_base

    @staticmethod
    def read_json(path_to_json):
        with open(path_to_json) as f:
            return json.load(f)

    @staticmethod
    def remove_duplicate(mylist):
        seen = set()
        newlist = []
        for item in mylist:
            t = tuple(item)
            if t[0] not in seen:
                newlist.append(item)
                seen.add(t[0])
        return newlist