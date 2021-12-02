import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import torch
import logging
import torchvision.models as models

from apex import amp

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from module.classification_package.src.utils import WarmupCosineSchedule
from module.classification_package.src.model import EmbeddingModel, Backbone
from module.classification_package.src.dataset import FishialDataset
from module.classification_package.src.dataset import BalancedBatchSampler
from module.classification_package.src.utils import find_device
from module.classification_package.src.loss_functions import TripletLoss, QuadrupletLoss
from module.classification_package.src.utils import NewPad
from module.classification_package.src.train import train
from module.pytorch_metric_learning import losses


# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)



def init_model(ckp=None):
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Identity()

    backbone = Backbone(resnet18)
    embedding_model = EmbeddingModel(backbone)
    if ckp:
        embedding_model.load_state_dict(torch.load(ckp))
    return embedding_model


def main():
    ds_train = FishialDataset(
        json_path="../dataset/data_train.json",
        root_folder="../dataset",
        transform=transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                  transforms.TrivialAugmentWide(),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.RandomErasing(p=0.358, scale=(0.05, 0.4), ratio=(0.05, 6.1), value=0, inplace=False),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    )

    ds_val = FishialDataset(
        json_path="../dataset/data_test.json",
        root_folder="../dataset",
        transform=transforms.Compose([
#             NewPad(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    device = find_device()
    writer = SummaryWriter('output/fashion_mnist_experiment_1')

    n_classes_per_batch = 12
    n_samples_per_class = 8

    balanced_batch_sampler_ds_train = BalancedBatchSampler(ds_train, n_classes_per_batch, n_samples_per_class)

    adaptive_margins = [True]
    learning_rates = [3e-2]
    momentums = [0.9]
    steps = 50000
    # batch_sizes = [32, 64, 128]

    # Tune hyperparams with val set.
    for adaptive_margin in adaptive_margins:
        for learning_rate in learning_rates:
            for momentum in momentums:
                data_loader_train = DataLoader(ds_train, batch_sampler=balanced_batch_sampler_ds_train,
                                              num_workers=2,
                                              pin_memory=True)  # Construct your Dataloader here

                model = init_model('output/ckpt_triplet_cross_entropy_0.845_50800.0.ckpt')     
                model.to(device)
                loss_fn = QuadrupletLoss()

                opt = torch.optim.SGD(model.parameters(),
                                                lr=learning_rate,
                                                momentum=momentum,
                                                weight_decay=0)
                
                scheduler =  WarmupCosineSchedule(opt, warmup_steps=500, t_total=steps) 
                model, opt = amp.initialize(models=model, optimizers=opt, opt_level='O2')
                amp._amp_state.loss_scalers[0]._loss_scale = 2**20
                # Convenient methods in order of verbosity from highest to lowest
                train(scheduler, steps, opt, model, data_loader_train, ds_val, device, writer, ['at_k'], loss_fn, logging)


if __name__ == '__main__':
    main()