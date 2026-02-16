import sys

# Change path specificly to your directories
sys.path.insert(1, '/home/andrew/Andrew/fish-identification')

import numpy as np
import torch

import random
import pyclipper
import cv2

from PIL import Image, ImageDraw
from shapely.geometry import Point
from shapely import geometry
from torch.utils.data.dataset import Dataset

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

        def _sanitize_polygon(coords):
            """
            Make polygon valid to avoid 'donut'/self-intersection artifacts from rasterization.
            Returns list of (x,y) tuples or None.
            """
            if not coords or len(coords) < 3:
                return None
            try:
                pts = [(float(p[0]), float(p[1])) for p in coords]
                poly = geometry.Polygon(pts)
                if poly.is_empty:
                    return None
                if not poly.is_valid:
                    poly = poly.buffer(0)  # fix self-intersections
                if poly.is_empty:
                    return None
                # If fixed polygon becomes MultiPolygon, pick the largest component
                if isinstance(poly, geometry.MultiPolygon):
                    poly = max(poly.geoms, key=lambda g: g.area, default=None)
                    if poly is None or poly.is_empty:
                        return None
                if not isinstance(poly, geometry.Polygon):
                    return None
                xs, ys = poly.exterior.coords.xy
                drawable = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
                # Clip to image bounds
                drawable = [(max(0, min(img_w - 1, x)), max(0, min(img_h - 1, y))) for x, y in drawable]
                return drawable if len(drawable) >= 3 else None
            except Exception:
                return None

        drawable_poly = _sanitize_polygon(poly_coords)
        if drawable_poly is None:
            mask_np = np.zeros((img_h, img_w), dtype=np.bool_)
        else:
            pil_mask = Image.new('1', (img_w, img_h), 0)
            try:
                ImageDraw.Draw(pil_mask).polygon(drawable_poly, outline=1, fill=1)
                mask_np = np.array(pil_mask, dtype=np.bool_)
            except Exception:
                mask_np = np.zeros((img_h, img_w), dtype=np.bool_)

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
        instance_data = {
            'drawn_fish_id': item_data['drawn_fish_id'],
            'annotation_id': item_data['annotation_id'],
            'image_id': item_data['image_id'],
        }

        img_path = item_data['file_name']
        polyline_initial = item_data.get('poly', [])
        
        try:
            pil_image_original = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image file not found: {img_path}")
            return torch.zeros((3, *self.img_size)), torch.tensor(-1), torch.zeros((1, *self.img_size)), instance_data

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
                return torch.zeros((3, *self.img_size)), torch.tensor(item_data['id_internal']), torch.zeros((1, *self.img_size)), instance_data

        if numeric_mask.ndim == 2:
            numeric_mask = numeric_mask.unsqueeze(0)

        object_mask_tensor = numeric_mask.float()

        label_tensor = torch.tensor(item_data['id_internal'])

        return image_tensor, label_tensor, object_mask_tensor, instance_data