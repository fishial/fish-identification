import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/fishial/Fishial/Object-Detection-Model')


# python Object-Detection-Model/helper/classification/CreateDataBaseTensor.py -c output/classification/resnet_18_186_train_09_11_not_ctop_poly/setup.yaml -m '/home/fishial/Fishial/dataset/data_for_deploy_poly' -a 'data_train.json' 'data_test.json' 'data_remain.json' 'data_out_of_class.json'

#python Object-Detection-Model/helper/classification/CreateDataBaseTensor.py -c output/classification/resnet_18_186_train_09_11_not_ctop_poly/setup.yaml -m '/home/fishial/Fishial/dataset/data_for_deploy_poly_fixed' -a 'data_train.json'
import os
import cv2
import yaml
import torch
import random
import argparse

from pathlib import Path
from PIL import Image
from torchvision import transforms

from module.classification_package.src.utils import read_json, save_json
from module.classification_package.src.model import init_model


def get_config(path):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def main():
    parser = argparse.ArgumentParser(description=' ')
    
    parser.add_argument("--config", "-c", required=True,
                        help="Path to the config yaml file")
        
    parser.add_argument("--main_folder", "-m", default = './', required=True,
                        help="Path to dataset directory")
    
    parser.add_argument("--annotation", "-a", required=True,
                        help="Path to annotation file", nargs='+', default=[])
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    absolute_path = Path(args.config).parent.absolute()
    
    config.update({
        'checkpoint': os.path.join(absolute_path, config['file_name'] + '.ckpt')
    })

    model = init_model(config)
    model.eval()

    loader = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    list_numbers = random.choices([100,100], k=config['model']['embeddings']) 
    random_numbers = torch.Tensor(list_numbers)
    
    for ann_path in args.annotation:
        name_ann = os.path.basename(ann_path)
        
        data_train = read_json(os.path.join(args.main_folder, ann_path))
        dict_info = {label: idx for idx, label in enumerate(set(data_train['label']))}
        data_set_ids = {idx: {
            'image_id':[],
            'annotation_id': []
        } for idx, label in enumerate(set(data_train['label']))}
        data_set = [[] for i in range(len(set(data_train['label'])))]

        for idx in range(len(data_train['label'])):
            ttotal = len(data_train['label'])
            print(f'Name: {ann_path} Left: {ttotal - idx}', end = '\r')
            img_path = os.path.join(args.main_folder, data_train['img_path'][idx])

            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)
            image = loader(image).float()
            image = torch.tensor(image)

            output = model(image.unsqueeze(0)).clone().detach()[0]
            
            data_set[dict_info[data_train['label'][idx]]].append(output)
            data_set_ids[dict_info[data_train['label'][idx]]]['annotation_id'].append(data_train['image_id'][idx])
            data_set_ids[dict_info[data_train['label'][idx]]]['image_id'].append(data_train['image_id_coco'][idx])

        dict_info = {idx: label for idx, label in enumerate(set(data_train['label']))}

        max_val = max(len(i) for i in data_set)
        for i in range(len(data_set)):
            if len(data_set[i]) < max_val:
                for _ in range(max_val - len(data_set[i])):
                    data_set[i].append(random_numbers)

        data_set = torch.stack ([torch.stack(i) for i in data_set] )
        torch.save(data_set, os.path.join(absolute_path, name_ann + '_embedding_dep_fixed.pt'))
        save_json(dict_info, os.path.join(absolute_path, name_ann + '_labels_dep_fixed.json'))
        save_json(data_set_ids, os.path.join(absolute_path, name_ann + '_idx_dep_fixed.json'))
    
if __name__ == '__main__':
    main()