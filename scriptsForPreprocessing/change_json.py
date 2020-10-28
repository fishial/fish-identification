import os, json
path_to_annotations = os.path.join(r'resources/fishial/train', 'via_region_data.json')
json_tmp = json.load(open(path_to_annotations))
result_dictionary = {}
for idx, name_record in enumerate(json_tmp):
    dict = json_tmp[name_record]
    dict.update({"verified": False})
    dict.update({"correct": False})
    result_dictionary.update({
        name_record: dict})
with open(path_to_annotations, 'w') as fp:
    json.dump(result_dictionary, fp)