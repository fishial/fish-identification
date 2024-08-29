import sys

# Change path specificly to your directories
sys.path.insert(1, '')
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


def save_conf(conf, path):
    with open(os.path.join(path, 'setup.yaml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)


def get_config(path):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)        

def main():
    parser = argparse.ArgumentParser(description='Embedding network train pipline.')
    parser.add_argument("--config", "-c", required=True,
                        help="Path to the congig yaml file")

    args = parser.parse_args()
    config = get_config(args.config)
    
    FO_DATASET_NAME_TRAIN = 'classification-v0.8.1_40_250_TRAIN'
    FO_DATASET_NAME_VALIDATION = 'classification-v0.8.1_40_250_VALIDATION'
    
    
    config['output_folder'] = os.path.join(
        config['output_folder'], 
        FO_DATASET_NAME_TRAIN, 
        config['train']['loss']['name'], 
        datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    )
    
    os.makedirs(config['output_folder'], exist_ok=True)
    
    fo_dataset = fo.load_dataset(FO_DATASET_NAME_TRAIN)
    train_data = fo_dataset.match_tags("train")
    train_val = fo_dataset.match_tags("val")
    
    fo_dataset_validation = fo.load_dataset(FO_DATASET_NAME_VALIDATION)
    
    validation_records = get_data_config(fo_dataset_validation)
    label_to_id_validation = {label:label_id for label_id, label in enumerate(list(validation_records))}
    
    train_records = get_data_config(train_data)
    val_records = get_data_config(train_val)
    
    label_to_id = {label:label_id for label_id, label in enumerate(list(train_records))}
    id_to_label = {label_id:label for label_id, label in enumerate(list(train_records))}
    
    save_json(id_to_label, os.path.join(config['output_folder'], 'labels.json'))
    
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
    
    extra_val = FishialDatasetFoOnlineCuting(
        validation_records,
        label_to_id_validation,
        transform=transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    crop_type = 'rect')
    print(f'extra_val.n_classes: {extra_val.n_classes}')
    

    if config['device'] is None:
        device = find_device()
    else:
        device = config['device']

    batch_size = config['dataset']['batchsampler']['classes_per_batch'] * config['dataset']['batchsampler']['samples_per_class']
    
    sampler = samplers.MPerClassSampler(ds_train.targets, m=config['dataset']['batchsampler']['samples_per_class']
                                        ,batch_size=batch_size, length_before_new_iter=len(ds_train))
    batch_sampler = BatchSampler(sampler, batch_size = batch_size, drop_last = False)

    data_loader_train = DataLoader(ds_train, batch_sampler=batch_sampler,
                                   num_workers=2,
                                   pin_memory=True)  # Construct your Dataloader here
    epoch = config['train']['epoch']
    model = init_model(ds_train.n_classes, embeddings = config['model']['embeddings'], backbone_name=config['model']['backbone'], checkpoint_path = config['checkpoint'], device = device)
    model.to(device)
    print(model)
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=f"{config['output_folder']}/app.log",                     # Имя файла для записи логов
                        filemode='w'                            # Режим записи в файл (w - перезаписывать файл, a - дописывать в конец файла)
    )



    if config['train']['loss']['name'] == 'quadruplet':
        loss_fn = QuadrupletLoss(config['train']['loss']['adaptive_margin'])
    elif config['train']['loss']['name'] == 'triplet':
        loss_fn = TripletLoss()
    elif config['train']['loss']['name'] == 'tripletohnm':
        loss_fn = WrapperOHNM()
    elif config['train']['loss']['name'] == 'angular':
        loss_fn = WrapperAngular()
    elif config['train']['loss']['name'] == 'pnploss':
        loss_fn = WrapperPNPLoss()
        
    opt = torch.optim.SGD(model.parameters(),
                          lr=config['train']['learning_rate'],
                          momentum=config['train']['momentum'],
                          weight_decay=0)

    scheduler = WarmupCosineSchedule(opt, warmup_steps=config['train']['warmup_steps'], t_total=epoch * len(data_loader_train))
    model, opt = amp.initialize(models=model, optimizers=opt, opt_level=config['train']['opt_level'])
    amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
    os.makedirs(config['output_folder'], exist_ok=True)
    save_conf(config, config['output_folder'])
    train(scheduler, epoch, opt, model, data_loader_train, ds_val, device, ['at_k'], loss_fn,
          logging,
          eval_every=20,
          file_name=config['file_name'],
          output_folder=config['output_folder'],
         extra_val = extra_val)
if __name__ == '__main__':
    main()