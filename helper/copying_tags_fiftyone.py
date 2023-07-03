import argparse
import fiftyone as fo
import fiftyone.zoo as foz
from tqdm import tqdm
import os

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--old_ds_name', type=str, default='fishial-dataset-november-2022')
    parser.add_argument('-dst','--new_ds_name', type=str, default='export-fishial-14-06-2023')
    parser.add_argument('-t','--tag', type=str, default='val')

    return parser

def main(args):

    DATASET_SRC = args.old_ds_name
    DATASET_DST = args.new_ds_name
    
    TAG = args.tag

    ds_src = fo.load_dataset(DATASET_SRC)
    ds_dst = fo.load_dataset(DATASET_DST)

    data_src = {}
    for sample in tqdm(ds_src):
        data_src.update({
            sample['image_id']: {
                'tags': sample['tags'],
                'polylines': sample['polylines']['polylines']
            }
        })

    for sample in tqdm(ds_dst):
        sample['tags'].append('train')
        if str(sample['image_id']) in data_src:
            if TAG in data_src[str(sample['image_id'])]['tags']:
                sample['tags'].remove('train')
                sample['tags'].append(TAG)
        sample.save()
    ds_dst.save()

if __name__ == '__main__':
    main(arg_parser().parse_args())