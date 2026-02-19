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
                continue
                # logging.warning(f"Skipping annotation {ann_inst['id']} due to missing keys.")
    
    def validate_image_file(self, filepath):
        """Validates that the file exists and is a valid image."""
        # Check if file exists
        if not os.path.exists(filepath):
            logging.error(f"File does not exist: {filepath}")
            return False
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(filepath):
            logging.error(f"Path is not a file: {filepath}")
            return False
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        _, ext = os.path.splitext(filepath.lower())
        if ext not in valid_extensions:
            logging.error(f"Invalid file extension '{ext}' for file: {filepath}")
            return False
        
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            logging.error(f"File is empty (0 bytes): {filepath}")
            return False
        
        # Try to read the image with cv2 to verify it's valid
        try:
            img = cv2.imread(filepath)
            if img is None:
                logging.error(f"cv2.imread returned None for file: {filepath}")
                return False
        except Exception as e:
            logging.error(f"Failed to read image with cv2: {filepath}, error: {e}")
            return False
        
        return True
    
    def create_dataset(self):
        """Creates the FiftyOne dataset from the processed data using labels."""
        samples = []
        skipped_files = []
        
        logging.info(f"Starting dataset creation with {len(self.images_info)} images")
        
        for idx, (image_id, info) in enumerate(self.images_info.items()):
            if not info['poly']:
                logging.debug(f"Skipping image {info['file_name']} - no polygons")
                continue
            
            filepath = info['file_name']
            logging.info(f"Processing image {idx+1}/{len(self.images_info)}: {filepath}")
            
            # Validate the image file
            if not self.validate_image_file(filepath):
                skipped_files.append(filepath)
                continue
            
            # Convert to absolute path if not already
            abs_filepath = os.path.abspath(filepath)
            logging.debug(f"Using absolute path: {abs_filepath}")
            
            dict_of_labels = {}
            for category, drawn_fish_id, ann_id, polylines in info['poly']:
                polyline = fo.Polyline(
                    label=category['supercategory'],
                    points=[polylines],
                    closed=False,
                    filled=True,
                    drawn_fish_id=drawn_fish_id,
                    ann_id=ann_id
                )
                
                if category['name'] in dict_of_labels:
                    dict_of_labels[category['name']].append(polyline)
                else:
                    dict_of_labels.update({category['name']: [polyline]})
            
            try:
                logging.debug(f"Creating FiftyOne sample for: {abs_filepath}")
                sample = fo.Sample(filepath=abs_filepath)
                
                # Check if media_type is valid
                if sample.media_type == 'unknown':
                    logging.error(f"Sample has unknown media_type for: {abs_filepath}")
                    skipped_files.append(filepath)
                    continue
                
                logging.debug(f"Sample media_type: {sample.media_type}")
                
                sample['image_id'] = image_id
                sample['width'] = info['width']
                sample['height'] = info['height']
                
                for label in dict_of_labels:
                    sample[label] = fo.Polylines(polylines=dict_of_labels[label])
                
                samples.append(sample)
                logging.debug(f"Successfully created sample for: {abs_filepath}")
            except Exception as e:
                logging.error(f"Failed to create sample for {abs_filepath}: {type(e).__name__}: {e}")
                skipped_files.append(filepath)
                continue
        
        logging.info(f"Created {len(samples)} valid samples")
        if skipped_files:
            logging.warning(f"Skipped {len(skipped_files)} files:")
            for skipped in skipped_files[:10]:  # Show first 10
                logging.warning(f"  - {skipped}")
            if len(skipped_files) > 10:
                logging.warning(f"  ... and {len(skipped_files) - 10} more")
        
        # Check if dataset already exists and delete it
        if fo.dataset_exists(self.dataset_name):
            logging.warning(f"Dataset '{self.dataset_name}' already exists. Deleting it...")
            fo.delete_dataset(self.dataset_name)
        
        # Create dataset and add samples
        logging.info(f"Creating FiftyOne dataset '{self.dataset_name}'")
        dataset = fo.Dataset(self.dataset_name)
        
        logging.info(f"Adding {len(samples)} samples to dataset...")
        dataset.add_samples(samples)
        dataset.persistent = True
        logging.info(f"Dataset created successfully with {len(samples)} samples!")

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
