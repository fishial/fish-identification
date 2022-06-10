import sys

# Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')
import os
import yaml
import torch
import logging
import argparse

from apex import amp

from PIL import Image
from torchvision import transforms

from torch.utils.data import DataLoader

from module.classification_package.src.utils import WarmupCosineSchedule
from module.classification_package.src.model import init_model
from module.classification_package.src.dataset import FishialDataset
from module.classification_package.src.dataset import BalancedBatchSampler
from module.classification_package.src.utils import find_device
from module.classification_package.src.loss_functions import TripletLoss, QuadrupletLoss, MultiSimilarityLoss
from module.classification_package.src.train import train

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def save_conf(conf, path):
    with open(os.path.join(path, 'setup.yaml'), 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)


def get_config(path):
    with open(path, "r") as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def main():
    parser = argparse.ArgumentParser(description='Embedding network train pipline.')
    parser.add_argument("--config", "-c", required=True,
                        help="Path to the congig yaml file")
    args = parser.parse_args()
    config = get_config(args.config)

    ds_train = FishialDataset(
        json_path=config['dataset']['train']['json_path'],
        root_folder=config['dataset']['train']['path'],
        transform=transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                      transforms.TrivialAugmentWide(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.RandomErasing(p=0.358, scale=(0.05, 0.4), ratio=(0.05, 6.1),
                                                               value=0, inplace=False),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    )

    ds_val = FishialDataset(
        json_path=config['dataset']['test']['json_path'],
        root_folder=config['dataset']['test']['path'],
        transform=transforms.Compose([
            #         NewPad(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    if config['device'] is None:
        device = find_device()
    else:
        device = config['device']

    balanced_batch_sampler_ds_train = BalancedBatchSampler(ds_train,
                                                           config['dataset']['batchsampler']['classes_per_batch'],
                                                           config['dataset']['batchsampler']['samples_per_class'])

    data_loader_train = DataLoader(ds_train, batch_sampler=balanced_batch_sampler_ds_train,
                                   num_workers=2,
                                   pin_memory=True)  # Construct your Dataloader here
    steps = len(data_loader_train) * config['train']['epoch']
    model = init_model(config)
    model.to(device)

    if config['train']['loss']['name'] == 'qudruplet':
        loss_fn = QuadrupletLoss(config['train']['loss']['adaptive_margin'])
    elif config['train']['loss']['name'] == 'triplet':
        loss_fn = TripletLoss('cuda')

    opt = torch.optim.SGD(model.parameters(),
                          lr=config['train']['learning_rate'],
                          momentum=config['train']['momentum'],
                          weight_decay=0)

    scheduler = WarmupCosineSchedule(opt, warmup_steps=config['train']['warmup_steps'], t_total=steps)
    model, opt = amp.initialize(models=model, optimizers=opt, opt_level=config['train']['opt_level'])
    amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
    os.makedirs(config['output_folder'], exist_ok=True)
    save_conf(config, config['output_folder'])
    train(scheduler, steps, opt, model, data_loader_train, ds_val, device, ['at_k'], loss_fn,
          logging,
          eval_every=len(data_loader_train),
          file_name=config['file_name'],
          output_folder=config['output_folder'])


if __name__ == '__main__':
    main()