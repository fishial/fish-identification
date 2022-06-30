import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')


# python helper/classification/vector_creator_by_ds.py -c output/train_results/resnet_18_98_finall_update/setup.yaml

import os
import yaml
import torch
import random
import argparse
from pathlib import Path
from torchvision import transforms

from module.classification_package.src.dataset import FishialDataset, FishialDatasetOnlineCuting
from module.classification_package.src.utils import read_json, save_json
from module.classification_package.src.model import init_model

def get_config(path):
    with open(path, "r") as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def main():
    parser = argparse.ArgumentParser(description=' ')

    parser.add_argument("--config", "-c", required=True,
                        help="Path to the config yaml file")


    args = parser.parse_args()

    config = get_config(args.config)
    absolute_path = Path(args.config).parent.absolute()

    config.update({
        'checkpoint': os.path.join(absolute_path, config['file_name'] + '.ckpt')
    })

    model = init_model(config)
    model.eval()

    list_numbers = random.choices([100,100], k=256)
    random_numbers = torch.Tensor(list_numbers)

    path_to_images_folder = r'datasets/fishial_collection_V2.0/FULL'
    path_to_COCO_file = r'../new_data_set/export_Verified_ALL_v2.json'

    for ann_name in ['train','test']:
        ds = FishialDatasetOnlineCuting(
            path_to_images_folder = path_to_images_folder,
            path_to_COCO_file = path_to_COCO_file,
            dataset_type = ann_name,
            transform=transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]))

        dict_info = ds.labels_dict
        print(dict_info)
        dict_idx = {ann_label: {
            'image_id': [],
            'annotation_id': []
        } for ann_label in range(ds.n_classes)}

        data_set = [[] for i in range(ds.n_classes)]

        for idx, rec in enumerate(ds):
            ttotal = len(ds)
            print(f'Name: {ann_name} Left: {ttotal - idx}', end = '\r')

            output = model(rec[0].unsqueeze(0)).clone().detach()[0]
            dict_idx[int(rec[1])]['annotation_id'].append(rec[2]['ann_id'])
            dict_idx[int(rec[1])]['image_id'].append(rec[2]['image_id'])
            data_set[int(rec[1])].append(output)
        max_val = max(len(i) for i in data_set)
        for i in range(len(data_set)):
            if len(data_set[i]) < max_val:
                for _ in range(max_val - len(data_set[i])):
                    data_set[i].append(random_numbers)
        data_set = torch.stack ([torch.stack(i) for i in data_set] )
        torch.save(data_set, os.path.join(absolute_path, ann_name + '_odm_embedding.pt'))
        save_json(dict_idx, os.path.join(absolute_path, ann_name + '_odm_idx.json'))
        save_json(dict_info, os.path.join(absolute_path, ann_name + '_odm_labels.json'))

if __name__ == '__main__':
    main()