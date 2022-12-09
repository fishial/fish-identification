import os
import json
import requests
import argparse

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from os import listdir
from os.path import isfile, join

def find_image_by_id(id: int):
    for i in data['images']:
        if i['id'] == id:
            return i
        
def read_json(data):
    with open(data) as f:
        return json.load(f)

def download(url):
    r = requests.get(url[0], allow_redirects=True)  # to get content after redirection
    with open(url[1], 'wb') as f:
        f.write(r.content)
    print("Current: {}".format(url[2]), end='\r')
    
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coco_path', type=str)
    parser.add_argument('-i','--image_path', type=str)

    return parser

#python Object-Detection-Model/helper/segmentation/downloader_coco_imgs.py -c '/home/fishial/Fishial/dataset/export/03_export_Verified_ALL.json.json' -i '/home/fishial/Fishial/dataset/fishial_collection-test'
def main(args):
    COCO_PATH = args.coco_path #'dataset/export/03_export_Verified_ALL.json'
    IMAGE_FOLDER = args.image_path #"dataset/fishial_collection"
    data = read_json(COCO_PATH)
    
    path_to_folder = "{}/data".format(IMAGE_FOLDER)
    os.makedirs(path_to_folder, exist_ok=True)
    imgs_files = [f for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]
    list_sd = []
    urls = []
    for i in tqdm(range(len(data['images']))):
        if data['images'][i]['file_name'] in imgs_files: continue

        list_sd.append(data['images'][i]['file_name'])
        folder_type = 'data'
        path = os.path.join(os.path.join(IMAGE_FOLDER, folder_type), data['images'][i]['file_name'])
        urls.append([data['images'][i]['coco_url'], path, i])

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download, urls) #urls=[list of url]

if __name__ == '__main__':
    main(arg_parser().parse_args())