import sys

# Change path specificly to your directories
sys.path.insert(1, '/home/andrew/Andrew/fish-identification')

import pandas as pd
import collections
import numpy as np
import os.path
import torch
import os

import random
import pyclipper
import time
import copy
import math
import cv2

from PIL import Image
from os import walk
from shapely.geometry import Polygon, Point
from shapely import geometry
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler
from module.segmentation_package.src.utils import get_mask
from module.classification_package.src.utils import read_json

import albumentations as A
from albumentations.pytorch import ToTensorV2


class FishialDatasetFoOnlineCuting(Dataset):
    def __init__(self,
                 records,
                 labels_dict,
                 train_state=False,
                 transform=None,
                 crop_type='poly',
                 background_fill_type=None, # 'black', 'gaussian_noise', 'random_color'
                 img_size=(224, 224)): # Added img_size for default Resize
        
        self.records_input = records # Keep original if needed, or deepcopy
        self.labels_dict = labels_dict
        self.train_state = train_state
        self.crop_type = crop_type
        self.background_fill_type = background_fill_type
        self.img_size = img_size
        
#         self.mask_cache_dir = os.path.join("mask_cache_not_fill", self.crop_type)
#         os.makedirs(self.mask_cache_dir, exist_ok=True)
        

        # Add internal id by dictionary
        # Create a new list of items instead of modifying records in-place potentially
        self.data_compleated = []
        for label, items_for_label in self.records_input.items():
            if label not in self.labels_dict:
                print(f"Warning: Label '{label}' not found in labels_dict. Skipping items for this label.")
                continue
            internal_id = self.labels_dict[label]
            for item_dict in items_for_label:
                # Make a copy to avoid modifying original 'records' dicts if they are reused
                new_item = item_dict.copy()
                new_item['id_internal'] = internal_id
                new_item['original_label'] = label # Keep original label if needed
                self.data_compleated.append(new_item)
        
        if transform is None:
            self.transform = A.Compose([
                A.Resize(*self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = transform
            
        if not self.data_compleated:
            print("Warning: Dataset is empty after processing records and labels_dict.")
            self.targets = []
            self.annotation_ids = []
            self.n_classes = 0
        else:
            self.targets = [item['id_internal'] for item in self.data_compleated]
            self.annotation_ids = [item['id'] for item in self.data_compleated] # Assuming 'id' is the annotation id
            # 'name' key was used for n_classes, ensure it exists or use 'original_label'
            # Using 'original_label' as derived from the keys of 'records'
            self.n_classes = len(set(item['original_label'] for item in self.data_compleated))


#     def __get_mask_path(self, item_data):
#         base_name = os.path.basename(item_data['file_name']).replace('.', '_')
#         annotation_id = item_data.get('id', 'unknown')
#         return os.path.join(self.mask_cache_dir, f"{annotation_id}_{base_name}.npy")


    def __get_margin(self, poly_coords):
        if len(poly_coords) < 3: # Need at least 3 points for a polygon
            return 0 # No margin if not a valid polygon
        
        try:
            poly_shapely = geometry.Polygon(poly_coords)
            if not poly_shapely.is_valid: # Check if polygon is valid
                # Attempt to buffer by 0 to fix invalid polygons like self-intersections
                poly_shapely = poly_shapely.buffer(0)
                if not poly_shapely.is_valid or poly_shapely.is_empty:
                    return 0 
            
            # Get minimum bounding box around polygon
            # minimum_rotated_rectangle might not be a Polygon if poly_shapely is just a LineString or Point after buffer(0)
            if isinstance(poly_shapely, geometry.Polygon) and not poly_shapely.is_empty:
                box = poly_shapely.minimum_rotated_rectangle
                if isinstance(box, geometry.LineString) or isinstance(box, geometry.Point) or box.is_empty:
                    # Fallback for degenerate cases: use envelope (axis-aligned bounding box)
                    min_x, min_y, max_x, max_y = poly_shapely.envelope.bounds
                    width = max_x - min_x
                    length = max_y - min_y
                else:
                    x, y = box.exterior.coords.xy
                    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])),
                                   Point(x[1], y[1]).distance(Point(x[2], y[2])))
                    length = max(edge_length)
                    width = min(edge_length)
            else: # Fallback if not a polygon (e.g. LineString, Point)
                min_x, min_y, max_x, max_y = poly_shapely.bounds
                width = max_x - min_x
                length = max_y - min_y

            if width == 0 or length == 0:
                return 0

            marg = int(min(width, length) * 0.04) # 4% of the smaller dimension
            random_margin = random.randint(-marg, int(marg * 1.9))
            return random_margin
        except Exception as e:
            # print(f"Warning: Could not calculate margin for polygon {poly_coords}. Error: {e}")
            return 0 # Default to no margin on error

    def __offset_polygon(self, poly_coords, offset_value, img_wh):
        """Offsets a polygon. Negative offset shrinks, positive expands."""
        if not poly_coords or offset_value == 0:
            return poly_coords

        pco = pyclipper.PyclipperOffset()
        # Pyclipper expects coordinates as integers or scaled floats
        # Ensure poly_coords are in the correct format, e.g., list of [x,y] lists or tuples
        scaled_poly = [(int(p[0]), int(p[1])) for p in poly_coords]
        pco.AddPath(scaled_poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        
        solution = pco.Execute(int(offset_value)) # Offset value should be an integer

        if not solution or not solution[0]: # If solution is empty or the first path is empty
            return None # Indicate failure

        offset_poly = solution[0]
        
        # Clip coordinates to image boundaries
        img_w, img_h = img_wh
        clipped_poly = []
        for p_x, p_y in offset_poly:
            clipped_x = max(0, min(img_w -1 , p_x))
            clipped_y = max(0, min(img_h -1 , p_y))
            clipped_poly.append((clipped_x, clipped_y))
        
        # Ensure it's still a valid polygon (at least 3 points)
        if len(clipped_poly) < 3:
            return None

        return clipped_poly

    def __create_masked_image(self, pil_img, poly_coords, img_path_for_log=""):
        original_img_np = np.array(pil_img)
        img_h, img_w = original_img_np.shape[:2]

        if not poly_coords or len(poly_coords) < 3:
            mask_np = np.zeros((img_h, img_w), dtype=np.bool_)
        else:
            pil_mask = Image.new('1', (img_w, img_h), 0)
            try:
                drawable_poly = [tuple(map(int, p)) for p in poly_coords]
                ImageDraw.Draw(pil_mask).polygon(drawable_poly, outline=1, fill=1)
                mask_np = np.array(pil_mask, dtype=np.bool_)
            except Exception:
                mask_np = np.ones((img_h, img_w), dtype=np.bool_)

        fill_type = self.background_fill_type
        if self.background_fill_type == 'random':
            fill_type = random.choice(['gaussian_noise', 'random_color', 'black', "None"])
            
        # Create background
        if fill_type == 'gaussian_noise':
            mean, std_dev = 128, 50
            noise = np.random.normal(mean, std_dev, original_img_np.shape)
            background_np = np.clip(noise, 0, 255).astype(np.uint8)
        elif fill_type == 'random_color':
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            background_np = np.full(original_img_np.shape, color, dtype=np.uint8)
        elif fill_type == 'black':
            background_np = np.zeros_like(original_img_np)
        else:
            return pil_img, mask_np

        final_img_np = np.where(mask_np[..., None], original_img_np, background_np)
        
        return Image.fromarray(final_img_np), mask_np

    def __len__(self):
        return len(self.data_compleated)
    
    def __getitem__(self, idx):
        
        item_data = self.data_compleated[idx]
        img_path = item_data['file_name']
        polyline_initial = item_data.get('poly', [])
        
        try:
            pil_image_original = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image file not found: {img_path}")
            return torch.zeros((3, *self.img_size)), torch.tensor(-1), torch.zeros((1, *self.img_size))

        processed_pil_image = pil_image_original.copy()

        current_polyline = polyline_initial
        if self.train_state and current_polyline and len(current_polyline) >= 3:
            img_wh = pil_image_original.size
            margin_offset = self.__get_margin(current_polyline)
            augmented_poly = self.__offset_polygon(current_polyline, margin_offset, img_wh)
            if augmented_poly:
                current_polyline = augmented_poly

        processed_pil_image, mask = self.__create_masked_image(pil_image_original, current_polyline, img_path)
        numeric_mask = mask.astype(np.uint8)

        # Ensure shapes match
        image_np = np.array(processed_pil_image)
        if image_np.shape[:2] != numeric_mask.shape:
            numeric_mask = cv2.resize(numeric_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.background_fill_type is None:
            processed_pil_image = pil_image_original  # без изменения фона

        if self.transform:
            try:
                transformed = self.transform(image=image_np, mask=numeric_mask)
                image_tensor = transformed["image"]
                numeric_mask = transformed["mask"]
            except Exception as e:
                print(f"ERROR: Failed to apply transform to image {img_path}. Error: {e}")
                return torch.zeros((3, *self.img_size)), torch.tensor(item_data['id_internal']), torch.zeros((1, *self.img_size))

        if numeric_mask.ndim == 2:
            numeric_mask = numeric_mask.unsqueeze(0)

        object_mask_tensor = numeric_mask.float()

        label_tensor = torch.tensor(item_data['id_internal'])

        return image_tensor, label_tensor, object_mask_tensor
    
# class FishialDatasetFoOnlineCuting(Dataset):
#     def __init__(self,
#                  records,
#                  labels_dict,
#                  train_state=False,
#                  transform=None,
#                  crop_type = 'poly'):
        
#         #Add internal id by dictionary
#         for label in records:
#             for k in records[label]:
#                 k.update({'id_internal': labels_dict[label]})
                
#         self.labels_dict = labels_dict
#         self.train_state = train_state
#         self.crop_type = crop_type
#         self.transform = transform
        
#         self.data_compleated = []
#         for label in records:
#             self.data_compleated.extend(records[label])
    
#         self.n_classes = len(set([i['name'] for i in self.data_compleated]))
#         self.targets = [i['id_internal'] for i in self.data_compleated]
#         self.annotation_ids = [i['id'] for i in self.data_compleated]
        
#     def __get_margin(self, poly):
#         # create example polygon
#         poly = geometry.Polygon(poly)

#         # get minimum bounding box around polygon
#         box = poly.minimum_rotated_rectangle

#         # get coordinates of polygon vertices
#         x, y = box.exterior.coords.xy

#         # get length of bounding box edges
#         edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

#         # get length of polygon as the longest edge of the bounding box
#         length = max(edge_length)

#         # get width of polygon as the shortest edge of the bounding box
#         width = min(edge_length)
#         marg = int(min(width, length) * 0.04)
#         random_margin = random.randint(-marg, int(marg * 1.9))
#         return random_margin

#     def __shrink_poly(self, poly, value, img_shape):
#         pco = pyclipper.PyclipperOffset()
#         pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
#         solution = pco.Execute(value)

#         for i in range(len(solution[0])):
#             solution[0][i][0] = max(0, min(img_shape[1], solution[0][i][0]))
#             solution[0][i][1] = max(0, min(img_shape[0], solution[0][i][1]))
#         return solution
    
#     def __get_poly_mask(self, img_path, polyline_main):
        
#         image = cv2.imread(img_path)
#         if self.train_state:
#             margine = self.__get_margin(polyline_main)
#             try:
#                 polyline_main = self.__shrink_poly(polyline_main, margine, image.shape[:2])[0]
#             except:
#                 print(f"Error: {img_path}")
#                 pass
#         mask = get_mask(image, np.array(polyline_main))
        
#         return mask
    
#     def __get_cropped_rect(self, img_path):

# #         img_path = self.data_compleated[idx]['file_name']
#         image = cv2.imread(img_path)
#         return image
    
#     def __len__(self):
#         # Return the length of the dataset
#         return len(self.data_compleated)
    
#     def __getitem__(self, idx):
#         # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
#         img_path = self.data_compleated[idx]['file_name']
#         polyline_main = self.data_compleated[idx]['poly']
        
#         if self.crop_type == 'poly':
#             image = self.__get_poly_mask(img_path, polyline_main)
#         else:
#             image = self.__get_cropped_rect(img_path)
            
#         mask = Image.fromarray(image) 
#         if self.transform:
#             mask = self.transform(mask)
        
#         return (mask, torch.tensor(self.data_compleated[idx]['id_internal']))
    
class FishialDatasetOnlineCuting(Dataset):
    def __init__(self,
                 path_to_images_folder,
                 path_to_COCO_file,
                 dataset_type='train',
                 train_state=False,
                 transform=None,
                 crop = False):

        min_image_per_class = 50
        min_eval_img = 15
        per_eval_img = 0.2
        max_img_per_class = 350

        self.path_to_images_folder = path_to_images_folder
        self.path_to_COCO_file = path_to_COCO_file
        self.dataset_type = dataset_type
        self.json_path = dataset_type
        self.train_state = train_state
        self.crop = crop

        data = read_json(path_to_COCO_file)
        filenames = next(walk(path_to_images_folder), (None, None, []))[2]  # [] if no file

        valid_images_indices = {}
        for image_rec_id, image_rec in enumerate(data['images']):
            if 'file_name' in data['images'][image_rec_id]:
                if data['images'][image_rec_id]['file_name'] not in filenames: continue
                if data['images'][image_rec_id]['fishial_extra']['xray'] or \
                        data['images'][image_rec_id]['fishial_extra']['not_a_real_fish'] or \
                        data['images'][image_rec_id]['fishial_extra']['no_fish'] or \
                        data['images'][image_rec_id]['fishial_extra']['test_image']: continue
                valid_images_indices.update({data['images'][image_rec_id]['id']: data['images'][image_rec_id]})
        list_of_valid_categories = {}

        for category in data['categories']:
            if category['name'] == 'General body shape' and category['supercategory'] != 'unknown':
                list_of_valid_categories.update({category['id']: {
                    'name': category['supercategory'],
                    'anns': {
                        'odm_true': [],
                        'odm_false': []
                    }
                }})

        for ann in data['annotations']:
            if 'category_id' not in ann: continue
            if not ann['is_valid']:
                print("isn't valid", end='\r')
                continue
            if ann['category_id'] in list_of_valid_categories:
                if ann['image_id'] in valid_images_indices:
                    ann.update({
                        'file_name': valid_images_indices[ann['image_id']]['file_name']
                    })
                    if len(ann['segmentation'][0]) < 10: continue
                    if valid_images_indices[ann['image_id']]['fishial_extra']['include_in_odm']:
                        list_of_valid_categories[ann['category_id']]['anns']['odm_true'].append(ann)
                    else:
                        list_of_valid_categories[ann['category_id']]['anns']['odm_false'].append(ann)
        list_of_valid_categories = dict(
            sorted(list_of_valid_categories.items(), key=lambda item: len(item[1]['anns']['odm_true']), reverse=True))
        main_class = copy.deepcopy(list_of_valid_categories)
        list_of_class_out_of_model = copy.deepcopy(list_of_valid_categories)

        for idx, i in enumerate(list_of_valid_categories):
            main_class[i]['anns']['odm_true'].extend(main_class[i]['anns']['odm_false'])
            list_of_class_out_of_model[i]['anns']['odm_true'].extend(list_of_class_out_of_model[i]['anns']['odm_false'])

            del main_class[i]['anns']['odm_false']
            del list_of_class_out_of_model[i]['anns']['odm_false']

            if len(main_class[i]['anns']['odm_true']) < min_image_per_class:
                del main_class[i]
            else:
                del list_of_class_out_of_model[i]

        for k_idx, k in enumerate(main_class):
            img_in_class = len(main_class[k]['anns']['odm_true'])

            eval_imgs_count = max(min_eval_img, min(max_img_per_class, img_in_class) * per_eval_img)
            main_class[k]['anns'].update({'test': []})
            main_class[k]['anns'].update({'train': []})
            main_class[k]['anns'].update({'remain': []})

            for z_idx, z in enumerate(main_class[k]['anns']['odm_true']):
                if z_idx > max_img_per_class:
                    main_class[k]['anns']['remain'].append(z)
                    continue

                if z_idx < eval_imgs_count:
                    main_class[k]['anns']['test'].append(z)
                else:
                    main_class[k]['anns']['train'].append(z)
            del main_class[k]['anns']['odm_true']

        self.data_compleated = []

        if self.dataset_type != 'out_of_class':
            for idx, i in enumerate(main_class):
                for ann in main_class[i]['anns'][self.dataset_type]:
                    self.data_compleated.append({
                        'name': main_class[i]['name'],
                        'id_fishial': i,
                        'id_internal': idx,
                        'image_id': ann['image_id'],
                        'ann_id': ann['id'],
                        'poly': [int(max(point, 0)) for point in ann['segmentation'][0]],
                        'file_name': ann['file_name']
                    })
        else:
            for idx, i in enumerate(list_of_class_out_of_model):
                for ann in list_of_class_out_of_model[i]['anns']['odm_true']:
                    self.data_compleated.append({
                        'name': list_of_class_out_of_model[i]['name'],
                        'id_fishial': i,
                        'id_internal': idx,
                        'image_id': ann['image_id'],
                        'ann_id': ann['id'],
                        'poly': [int(max(point, 0)) for point in ann['segmentation'][0]],
                        'file_name': ann['file_name']
                    })
        # clean memory
        main_class.clear()
        list_of_valid_categories.clear()
        valid_images_indices.clear()
        list_of_class_out_of_model.clear()

        self.transform = transform
        self.n_classes = len(set([i['name'] for i in self.data_compleated]))
        self.targets = [i['id_internal'] for i in self.data_compleated]
        self.labels_dict = {}
        for rec in range(len(self.data_compleated)):
            if self.data_compleated[rec]['id_internal'] not in self.labels_dict:
                self.labels_dict.update({
                    self.data_compleated[rec]['id_internal']: self.data_compleated[rec]['name']
                })

    def __get_margin(self, poly):
        # create example polygon
        poly = geometry.Polygon(poly)

        # get minimum bounding box around polygon
        box = poly.minimum_rotated_rectangle

        # get coordinates of polygon vertices
        x, y = box.exterior.coords.xy

        # get length of bounding box edges
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

        # get length of polygon as the longest edge of the bounding box
        length = max(edge_length)

        # get width of polygon as the shortest edge of the bounding box
        width = min(edge_length)
        marg = int(min(width, length) * 0.05)
        random_margin = random.randint(-marg, int(marg * 1.9))
        return random_margin

    def __shrink_poly(self, poly, value, img_shape):
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(value)

        for i in range(len(solution[0])):
            solution[0][i][0] = max(0, min(img_shape[1], solution[0][i][0]))
            solution[0][i][1] = max(0, min(img_shape[0], solution[0][i][1]))
        return solution
    
    def __get_poly_mask(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.path_to_images_folder, self.data_compleated[idx]['file_name'])
        poly_raw = self.data_compleated[idx]['poly']
        polyline_main = [[int(poly_raw[point_id * 2]), int(poly_raw[point_id * 2 + 1])] for point_id in
                         range(int(len(poly_raw) / 2))]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.dataset_type == 'train' and self.train_state:
            margine = self.__get_margin(polyline_main)
            polyline_main = self.__shrink_poly(polyline_main, margine, image.shape[:2])[0]
        mask = get_mask(image, np.array(polyline_main))

        if self.transform:
            mask = Image.fromarray(mask)
            mask = self.transform(mask)
        del image
        return mask
    
    def __get_cropped_mask(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.path_to_images_folder, self.data_compleated[idx]['file_name'])
        poly_raw = self.data_compleated[idx]['poly']
        polyline_main = [[int(poly_raw[point_id * 2]), int(poly_raw[point_id * 2 + 1])] for point_id in
                         range(int(len(poly_raw) / 2))]
                    
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                         
        rect = cv2.boundingRect(np.array(polyline_main))
        x, y, w, h = rect
        mask = image[y:y + h, x:x + w].copy()
                         
        if self.transform:
            mask = Image.fromarray(mask)
            mask = self.transform(mask)
        del image
        return mask
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.data_compleated)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if self.crop:
            mask = self.__get_poly_mask(idx)
        else:
            mask = self.__get_cropped_mask(idx)
        
        return (mask, torch.tensor(self.data_compleated[idx]['id_internal']))#, self.data_compleated[idx])


class FishialDataset(Dataset):
    def __init__(self, json_path, root_folder, transform=None):
        self.json_path = os.path.join(root_folder, json_path)
        self.data_frame = pd.DataFrame.from_dict(read_json(self.json_path))
        self.root_folder = root_folder

        self.__remove_unexists_imgs()
        self.transform = transform

        self.targets = [int(i) for i in self.data_frame['label_encoded'].tolist()]

        self.new_list = self.data_frame['label_encoded'].unique()

        def recall(label, dict_new):
            return [i for i, x in enumerate(dict_new) if x == label][0]

        self.data_frame['target'] = self.data_frame.apply(lambda x: recall(x['label_encoded'], self.new_list), axis=1)
        self.n_classes = len(self.new_list)

        self.library_name = {}
        for idx, z in enumerate(self.new_list):
            self.library_name.update(
                {
                    idx: {
                        "num": z,
                        'label': self.data_frame.loc[self.data_frame['label_encoded'] == z].iloc[0]['label']
                    }
                }
            )

    def __remove_unexists_imgs(self):
        for i in range(len(self.data_frame['image_id']) - 1, -1, -1):
            exist = os.path.isfile(os.path.join(self.root_folder, self.data_frame['img_path'][i]))
            if not exist:
                self.data_frame = self.data_frame.drop(self.data_frame.index[[i]])
        self.data_frame = self.data_frame.reset_index(drop=True)

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_folder, self.data_frame.iloc[idx]['img_path'])
        image = Image.open(img_name)
        class_id = self.data_frame.iloc[idx]['target']

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(class_id))


class BalancedBatchSampler(BatchSampler):
    """
    Returns batches of size n_classes * n_samples
    """
    
    def __init__(self, labels_list, n_classes, n_samples):
        
        self.labels_list = labels_list
            
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.batch_size = self.n_samples * self.n_classes
        
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.labels_list):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[ class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples
            
    def __len__(self):
        return len(self.labels_list) // self.batch_size