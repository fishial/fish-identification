import os, json
from shutil import copyfile
import random


def merge_data_set(data, proportions, data_set_name):
    array_with_data = []

    for i in data:
        path_to_annotations = os.path.join(i, 'via_region_data.json')
        json_tmp = json.load(open(path_to_annotations))
        for name_record in json_tmp:
            array_with_data.append([json_tmp[name_record], i])
    random.shuffle(array_with_data)

    for idx in range(len(proportions)):
        folder_path = os.path.join(data_set_name, proportions[idx][0])
        os.makedirs(folder_path, exist_ok=True)
        proportions[idx][1] = int(len(array_with_data) * proportions[idx][1] / 100)

    for folder in proportions:
        current_folder = os.path.join(data_set_name, folder[0])
        result_dict = {}
        print("Score: ", len(array_with_data), print(folder[1]))
        for idx_record in range(folder[1], -1, -1):
            if len(array_with_data) == 0: continue
            src_path = os.path.join(array_with_data[0][1], array_with_data[0][0]['filename'])
            dst_path = os.path.join(current_folder, array_with_data[0][0]['filename'])
            copyfile(src_path, dst_path)
            result_dict.update({array_with_data[0][0]['filename'] + str(idx_record): array_with_data[0][0]})
            array_with_data.pop(0)
        print("Dict: ", len(result_dict))
        with open(os.path.join(current_folder, 'via_region_data.json'), 'w') as fp:
            json.dump(result_dict, fp)


merge_data_set([r'fishial/train', r'parsing'],
               [['train', 100]], "fishial-new")
