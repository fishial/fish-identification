import sys
import os
# Modify sys.path to include the root directory containing 'fish-identification'
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])
    print("SETUP: sys.path updated")
    
import os
import yaml
import torch
import logging
import argparse

from datetime import datetime

import torchvision.models as models

from apex import amp

from PIL import Image
from torchvision import transforms

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_metric_learning import losses, miners, trainers, samplers
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data.sampler import BatchSampler

from module.classification_package.src.utils import WarmupCosineSchedule
from module.classification_package.src.model import init_model
from module.classification_package.src.dataset import FishialDatasetFoOnlineCuting
from module.classification_package.src.dataset import BalancedBatchSampler
from module.classification_package.src.utils import find_device
from module.classification_package.src.loss_functions import *
from module.classification_package.src.utils import NewPad
from module.classification_package.src.utils import get_data_config
from module.classification_package.src.train import train
from module.classification_package.src.utils import read_json, save_json

import fiftyone as fo
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


device = find_device()

epoch = 500
n_classes_per_batch = 30
n_samples_per_class = 5
dataset_fo = 'my_voxel_dataset_name_train'
output_folder = '/home/fishial/Fishial/output/classification_asa'
checkpoint = r'/home/fishial/Fishial/TEST_PIPLINE/classification/best_ckpt_0.6156.ckpt'

fo_dataset = fo.load_dataset(dataset_fo)
train_data = fo_dataset.match_tags("train")
train_val = fo_dataset.match_tags("val")

train_records = get_data_config(train_data)
val_records = get_data_config(train_val)

label_to_id = {label:label_id for label_id, label in enumerate(list(train_records))}
id_to_label = {label_id:label for label_id, label in enumerate(list(train_records))}
    
output_folder = os.path.join(
        output_folder, 
        dataset_fo, 
        'cross_entropy', 
        datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    )
    
os.makedirs(output_folder, exist_ok=True)
    
save_json(id_to_label,os.path.join(output_folder, 'labels.json'))
    
ds_train = FishialDatasetFoOnlineCuting(
        train_records,
        label_to_id,
        train_state=True,
        transform=transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                      transforms.RandomAutocontrast(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.RandomErasing(p=0.358, scale=(0.05, 0.4), ratio=(0.05, 6.1),
                                                               value=0, inplace=False),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    crop_type = 'rect')

print(f'ds_train.n_classes: {ds_train.n_classes}')

ds_val = FishialDatasetFoOnlineCuting(
    val_records,
    label_to_id,
    transform=transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    crop_type = 'rect')
print(f'ds_val.n_classes: {ds_val.n_classes}')

batch_size = n_classes_per_batch * n_samples_per_class

sampler = samplers.MPerClassSampler(ds_train.targets, m=n_samples_per_class
                                    ,batch_size=batch_size, length_before_new_iter=len(ds_train))
batch_sampler = BatchSampler(sampler, batch_size = batch_size, drop_last = False)

data_loader_train = DataLoader(ds_train, batch_sampler=batch_sampler,
                               num_workers=4,
                               pin_memory=True)  # Construct your Dataloader here

n_classes = ds_val.n_classes

model = init_model(n_classes, embeddings = 128, backbone_name='convnext_tiny', checkpoint_path = checkpoint, device = device)
model.to(device)
    
loss_fn = nn.CrossEntropyLoss()

opt = torch.optim.SGD(model.parameters(),
                      lr=3e-2,
                      momentum=0.9,
                      weight_decay=0)

scheduler = WarmupCosineSchedule(opt, warmup_steps=500, t_total=epoch * len(data_loader_train))
model, opt = amp.initialize(models=model, optimizers=opt, opt_level='O2')

amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
print(20 * "-")
train(scheduler, epoch, opt, model, data_loader_train, ds_val, device, ['accuracy'], loss_fn,
      logging,
      eval_every_epochs=5,
      output_folder=output_folder)
