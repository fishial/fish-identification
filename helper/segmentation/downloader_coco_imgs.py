import os
import json
import requests

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def find_image_by_id(id: int):
    for i in data['images']:
        if i['id'] == id:
            return i


f = open('data/export.json',)
data = json.load(f)
f.close()

def download(url):
    r = requests.get(url[0], allow_redirects=True)  # to get content after redirection
    with open(url[1], 'wb') as f:
        f.write(r.content)
    print("Current: {}".format(url[2]), end='\r')

folder_name = "fishial_collection"
os.makedirs(folder_name, exist_ok=True)
os.makedirs("{}/data".format(folder_name), exist_ok=True)
list_sd = []
urls = []
for i in tqdm(range(len(data['images']))):
    # if 'train_data' not in data['images'][i]:
    #     continue
    # print("tut")
    list_sd.append(data['images'][i]['file_name'])
    folder_type = 'data'
    path = os.path.join(os.path.join(folder_name, folder_type), data['images'][i]['file_name'])
    urls.append([data['images'][i]['coco_url'], path, i])

with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(download, urls) #urls=[list of url]
