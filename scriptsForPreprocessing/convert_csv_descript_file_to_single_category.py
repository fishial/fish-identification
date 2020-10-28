import os, json
import pandas as pd

name_csv = 'validation-annotations-object-segmentation.csv'
path_to_folder = r'resources'
path_to_csv_file = os.path.join(path_to_folder, name_csv)

df = pd.read_csv (path_to_csv_file)
print("Preview count record: ", len(df.index))
df = df[(df.LabelName == "/m/0ch_cf") | (df.LabelName == "/m/03fj2") | (df.LabelName == "/m/0by6g") | (df.LabelName == "/m/0m53l" ) | (df.LabelName == "/m/0nybt" ) | ( df.LabelName == "/m/0fbdv")]
df.reset_index(inplace=True)
del df['index']
print("Post process count record: ", len(df.index))
title, ext = os.path.splitext(os.path.basename(path_to_csv_file))
df.to_csv(os.path.join(path_to_folder, title + "-fish" + ext), encoding='utf-8', index=False)
