import os, json

# import data
path_to_json = r'resources/fishial/val/via_region_data.json'

json_tmp = json.load(open(path_to_json))
unique_img_array = []
json_tmp_clone = json_tmp.copy()
cnt_bad = 0

print(len(json_tmp))

# for key, v in json_tmp.items():
#     if len(v["regions"]['0']['shape_attributes']['all_points_x']) < 40:
#         del json_tmp_clone[key]
#         cnt_bad += 1
#         print('I find: ', len(v["regions"]['0']['shape_attributes']['all_points_x']))

for i in json_tmp:
    if json_tmp[i]['verified'] and not json_tmp[i]['correct']:
        cnt_bad +=1
        del json_tmp_clone[i]
print("Removed: {}".format(cnt_bad))
print("Wait for save ...")
with open(path_to_json, 'w') as fp:
    json.dump(json_tmp_clone, fp)
print("Change saved ! ")