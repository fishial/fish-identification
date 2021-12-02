import pandas as pd
import numpy as np
import os.path
import torch
import os

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler
from module.classification_package.src.utils import read_json


class FishialDataset(Dataset):
    def __init__(self, json_path, root_folder, transform=None):
        self.json_path = os.path.join(root_folder, json_path)
        self.data_frame = pd.DataFrame.from_dict(read_json(self.json_path))
        self.root_folder = root_folder
        
        self.__remove_unexists_imgs()
        self.transform = transform
        
        self.targets = [int(i) for i in self.data_frame['label_encoded'].tolist()]

        self.new_list = self.data_frame['label_encoded'].unique()

        def recall(label, dict_new):
            return [i for i, x in enumerate(dict_new) if x == label][0]

        self.data_frame['target'] = self.data_frame.apply(lambda x: recall(x['label_encoded'], self.new_list), axis=1)
        self.n_classes = len(self.new_list)

        self.library_name = {}
        for idx, z in enumerate(self.new_list):
            self.library_name.update(
                {
                    idx: {
                        "num": z,
                        'label': self.data_frame.loc[self.data_frame['label_encoded'] == z].iloc[0]['label']
                    }
                }
            )

    def __remove_unexists_imgs(self):
        for i in range(len(self.data_frame['image_id']) -1, -1, -1):
            exist = os.path.isfile(os.path.join(self.root_folder, self.data_frame['img_path'][i]))
            if not exist:
                self.data_frame = self.data_frame.drop(self.data_frame.index[[i]])
        self.data_frame = self.data_frame.reset_index(drop=True)
        

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_folder, self.data_frame.iloc[idx]['img_path'])
        image = Image.open(img_name)
        class_id = self.data_frame.iloc[idx]['target']

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(class_id))


class BalancedBatchSampler(BatchSampler):
    """
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
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