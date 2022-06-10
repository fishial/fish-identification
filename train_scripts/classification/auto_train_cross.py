# coding=utf-8
from __future__ import absolute_import, division, print_function

import sys

# Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

from module.classification_package.src.model import FcNet
from module.classification_package.src.model import Backbone
from module.classification_package.src.train import train

import torch
import torchvision.models as models
import logging

from apex import amp
from PIL import Image

from module.classification_package.src.utils import find_device
from module.classification_package.src.utils import WarmupCosineSchedule
from module.classification_package.src.dataset import BalancedBatchSampler
from module.classification_package.src.dataset import FishialDataset

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torchvision import transforms

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

writer = SummaryWriter('output/fashion_mnist_experiment_1')
device = find_device()

n_classes_per_batch = 2
n_samples_per_class = 5

ds_train = FishialDataset(
    json_path="data_train.json",
    root_folder="/home/codahead/Fishial/FishialReaserch/datasets/cutted_v2.5/data_set",
    transform=transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                  transforms.TrivialAugmentWide(),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.RandomErasing(p=0.358, scale=(0.05, 0.4), ratio=(0.05, 6.1), value=0,
                                                           inplace=False),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
)

ds_val = FishialDataset(
    json_path="data_test.json",
    root_folder="/home/codahead/Fishial/FishialReaserch/datasets/cutted_v2.5/data_set",
    transform=transforms.Compose([
        #         NewPad(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)

balanced_batch_sampler_ds_train = BalancedBatchSampler(ds_train, n_classes_per_batch, n_samples_per_class)

data_loader_train = DataLoader(ds_train, batch_sampler=balanced_batch_sampler_ds_train,
                               num_workers=2,
                               pin_memory=True)

ckp = None
n_classes = ds_val.n_classes

resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()

backbone = Backbone(resnet18)
model = FcNet(backbone, n_classes)
if ckp:
    model.load_state_dict(torch.load(ckp))
model.to(device)

loss_fn = nn.CrossEntropyLoss()
# opt = Adadelta(model.parameters(), lr=0.001)
opt = torch.optim.SGD(model.parameters(),
                      lr=3e-2,
                      momentum=0.9,
                      weight_decay=0)
epoch = 800
steps = len(data_loader_train) * epoch
scheduler = WarmupCosineSchedule(opt, warmup_steps=500, t_total=steps)
model, opt = amp.initialize(models=model,
                            optimizers=opt,
                            opt_level='O2')
amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
# Convenient methods in order of verbosity from highest to lowest
# train(scheduler, steps, opt, model, data_loader_train, ds_val, device, ['at_k'], loss_fn, logging, eval_every=len(data_loader_train))
train(scheduler, steps, opt, model, data_loader_train, ds_val, device, ['accuracy'], loss_fn, logger, eval_every=100)