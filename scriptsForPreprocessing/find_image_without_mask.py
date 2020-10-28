import os, json
import shutil
# import data
path_to_json = r'resources/fishial/train/via_region_data.json'
path_to_image = r'resources/fishial/train'

list_path_img = [os.path.basename(i) for i in os.listdir(path_to_image)]
print(len(list_path_img))
json_tmp = json.load(open(path_to_json))

list_of_using = []

for i in json_tmp:
    list_of_using.append(json_tmp[i]['filename'])
print("Before: {}".format(len(list_of_using)))
list_of_using = set(list_of_using)

for i in list_path_img:
    if i not in list_of_using:
        print(i)
        if i == 'via_region_data.json': continue
        dst = os.path.join(r'resources/img_free', i)
        src = os.path.join(path_to_image, i)
        shutil.move(os.path.join(path_to_image, i), r'resources/img_free')