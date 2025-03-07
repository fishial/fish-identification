import argparse
import json
import os
import cv2
import numpy as np
import logging
import multiprocessing
import fiftyone as fo
import fiftyone.zoo as foz

def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

setup_logging()

class CocoToVoxelConverter:
    def __init__(self, coco_path, image_path, dataset_name, skip_tags):
        self.coco_path = coco_path
        self.image_path = image_path
        self.dataset_name = dataset_name
        self.skip_tags = set(skip_tags)
        self.images_info = {}
        self.categories = {}
        self.data_coco_export = {}
    
    def read_json(self):
        """Reads the COCO JSON file."""
        try:
            with open(self.coco_path, 'r') as f:
                self.data_coco_export = json.load(f)
            logging.info("COCO JSON file loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to read JSON file: {e}")
            raise
    
    def load_categories(self):
        """Loads category information from COCO data."""
        self.categories = {
            cat['id']: {'name': cat['name'], 'supercategory': cat['supercategory'], 'species_id': cat.get('species_id', None)}
            for cat in self.data_coco_export.get('categories', [])
        }

    def get_files_in_folder(self):
        """Returns a set of file names in the given folder for fast lookup."""
        return {f for f in os.listdir(self.image_path) if os.path.isfile(os.path.join(self.image_path, f))}
    
    def check_key(self, instance, key):
        """Checks if a key exists in a dictionary and returns its value, otherwise returns True."""
        return instance.get(key, True)
    
    def clamp(self, value, min_val, max_val):
        """Clamps a value between a minimum and maximum value."""
        return max(min_val, min(value, max_val))
    
    def fix_poly(self, poly, shape):
        """Ensures polygon points are within image boundaries."""
        return [(self.clamp(p[0], 0, shape[0]), self.clamp(p[1], 0, shape[1])) for p in poly]
    
    def process_image(self, image_inst):
        """Processes an image entry from the COCO JSON file."""
        file_name = image_inst['file_name']
        if file_name not in self.image_files:
            return None
        
        if any(self.check_key(image_inst.get('fishial_extra', {}), key) for key in self.skip_tags):
            return None
        
        image_path = os.path.join(self.image_path, file_name)
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Could not read image: {image_path}")
            return None
        
        height, width = image.shape[:2]
        return {
            'id': image_inst['id'],
            'file_name': image_path,
            'width': width,
            'height': height,
            'include_in_odm': image_inst['fishial_extra'].get('include_in_odm', False),
            'poly': []
        }
    
    def process_annotations(self):
        """Processes all annotations from the COCO JSON file."""
        for ann_inst in self.data_coco_export['annotations']:
            image_id = ann_inst['image_id']
            if image_id not in self.images_info:
                continue
            
            try:
                drawn_fish_id = ann_inst['fishial_extra']['drawn_fish_id']
                segmentation = ann_inst['segmentation'][0]
                poly = [(int(segmentation[i * 2]), int(segmentation[i * 2 + 1])) for i in range(len(segmentation) // 2)]
                
                poly = self.fix_poly(poly, [self.images_info[image_id]['width'], self.images_info[image_id]['height']])
                poly = [[p[0] / self.images_info[image_id]['width'], p[1] / self.images_info[image_id]['height']] for p in poly]
                
                self.images_info[image_id]['poly'].append([
                    self.categories[ann_inst['category_id']], drawn_fish_id, ann_inst['id'], poly
                ])
            except KeyError:
                logging.warning(f"Skipping annotation {ann_inst['id']} due to missing keys.")
    
    def create_dataset(self):
        """Creates the FiftyOne dataset from the processed data using labels."""
        samples = []
        for image_id, info in self.images_info.items():
            if not info['poly']:
                continue
            
            dict_of_labels = {}
            for category, drawn_fish_id, ann_id, polylines in info['poly']:
                polyline = fo.Polyline(
                    label=category['supercategory'],
                    points=[polylines],
                    closed=False,
                    filled=True
                )
                
                if category['name'] in dict_of_labels:
                    dict_of_labels[category['name']].append(polyline)
                else:
                    dict_of_labels.update({category['name']: [polyline]})
            
            sample = fo.Sample(filepath=info['file_name'])
            sample['image_id'] = image_id
            sample['width'] = info['width']
            sample['height'] = info['height']
            
            for label in dict_of_labels:
                sample[label] = fo.Polylines(polylines=dict_of_labels[label])
            
            samples.append(sample)
        
        dataset = fo.Dataset(self.dataset_name)
        dataset.add_samples(samples)
        dataset.persistent = True
        logging.info("Dataset created successfully!")

    def run(self):
        """Executes the entire processing pipeline."""
        self.read_json()
        self.load_categories()
        self.image_files = self.get_files_in_folder()
        
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(self.process_image, self.data_coco_export['images'])
            self.images_info = {res['id']: res for res in results if res}
        
        self.process_annotations()
        self.create_dataset()
        logging.info("Processing completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coco_path', required=True, type=str, help="Path to COCO JSON file")
    parser.add_argument('-i', '--image_path', required=True, type=str, help="Path to image directory")
    parser.add_argument('-ds', '--data_set', required=True, type=str, help="Dataset name")
    parser.add_argument('-st', '--skip_tags', nargs='*', default=['xray', 'not_a_real_fish', 'no_fish'], help="List of tags to skip images")
    args = parser.parse_args()
    
    converter = CocoToVoxelConverter(args.coco_path, args.image_path, args.data_set, args.skip_tags)
    converter.run()
