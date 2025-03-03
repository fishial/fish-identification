import os
import numpy as np
import requests
import cv2
import logging
import base64
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LinearRing, MultiPolygon

# ============== CONFIGURATION ==============
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO)


# ============== POLYGON UTILITIES ==============

def compute_iou(poly1, poly2):
    """Compute IoU (Intersection over Union) between two polygons."""
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0


def max_iou(polygons, target_polygon):
    """Find the polygon with the highest IoU relative to the target polygon."""
    max_iou_value, best_polygon = 0, None
    for poly_id, poly in enumerate(polygons):
        iou = compute_iou(poly, target_polygon)
        if iou > max_iou_value:
            max_iou_value, best_polygon = iou, poly_id
    return max_iou_value, best_polygon


def bitmap_to_polygon(bitmap):
    """Convert a binary mask to polygon contours."""
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2:]

    if hierarchy is None:
        return [], False

    return sorted([c.reshape(-1, 2) for c in contours], key=len, reverse=True)


def poly_array_to_dict(polygon):
    """Convert a polygon array to a dictionary with labeled coordinates."""
    return {f"{axis}{i + 1}": int(coord) for i, (coord, axis) in enumerate(zip(polygon.flatten(), "xy" * len(polygon)))}


# ============== CONTOUR FIXING UTILITIES ==============

def is_contour_valid(contour):
    """Check if a contour is valid (no self-intersections)."""
    if len(contour) < 3:
        return False

    polygon = Polygon(contour)
    return polygon.is_valid and LinearRing(contour).is_simple


def fix_contour(contour):
    """Attempt to fix a self-intersecting contour using buffer(0)."""
    polygon = Polygon(contour)
    if polygon.is_valid:
        return contour

    fixed_polygon = polygon.buffer(0)

    if fixed_polygon.is_empty:
        return np.array([])

    return np.array(fixed_polygon.exterior.coords) if isinstance(fixed_polygon, Polygon) else \
        np.array(max(fixed_polygon.geoms, key=lambda p: p.area).exterior.coords)


def full_fix_contour(poly):
    """Validate and fix a contour if necessary."""
    if not poly or len(poly[0]) < 10:
        return [], "Empty Contour"

    contour = poly[0]
    if is_contour_valid(contour):
        return contour, None

    fixed_contour = fix_contour(contour)
    return (fixed_contour, "Fixed Contour") if fixed_contour.size > 0 and is_contour_valid(fixed_contour) else ([], "Can't fix")


# ============== IMAGE & MASK UTILITIES ==============

def resize_logits_mask_pil(logits_mask, width, height):
    """Resize a logits mask using PIL."""
    return np.array(Image.fromarray(logits_mask.astype(np.float32)).resize((width, height), Image.BILINEAR))


def resize_mask(mask, width, height):
    """Resize a binary mask using OpenCV (faster than scipy.ndimage.zoom)."""
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def create_mask(polygon, height, width, color=0):
    """Create a mask from a polygon."""
    scaled_polygon = [np.array([[p[0] * (width if max(max(polygon)) < 1.0 else 1),
                                 p[1] * (height if max(max(polygon)) < 1.0 else 1)] for p in polygon], np.int32)]

    mask = np.zeros((height, width, 3 if isinstance(color, tuple) else 1), dtype=np.uint8)
    cv2.fillPoly(mask, scaled_polygon, color)
    return mask


def resize_image(pil_img, block_size=32, min_div=4, max_div=25, hard_size=None):
    """Resize an image to the nearest multiple of `block_size`."""
    width, height = pil_img.size

    if hard_size:
        new_width, new_height = hard_size, hard_size
    else:
        new_width = nearest_multiple(width, block_size, min_div, max_div)
        new_height = nearest_multiple(height, block_size, min_div, max_div)

    return pil_img.resize((new_width, new_height), Image.LANCZOS)


def nearest_multiple(x, base, min_mult, max_mult):
    """Find the nearest multiple of `base` within a given range."""
    lower = base * max(min_mult, x // base)
    upper = base * min(max_mult, (x // base) + 1)
    return lower if x - lower <= upper - x else upper


# ============== DATA VISUALIZATION ==============

def visualize(**images):
    """Plot images side by side."""
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, len(images), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace("_", " ").title())
        plt.imshow(image)
    plt.show()


# ============== RANDOM IMAGE GENERATION ==============

def generate_random_image(max_width=2000, min_width=64, max_height=800, min_height=64):
    """Generate a random image of variable size."""
    width, height = np.random.randint(min_width, max_width), np.random.randint(min_height, max_height)
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def scale_polygon(vertices, x_ratio, y_ratio):
    """Scale a polygon by given x and y ratios."""
    return [(int(x * x_ratio), int(y * y_ratio)) for x, y in vertices]


def draw_polygon(pil_image, polygon_points, line_color=(0, 0, 255), line_width=5):
    """Draw a polygon on an image."""
    ImageDraw.Draw(pil_image).polygon(polygon_points, outline=line_color, width=line_width)


# ============== IMAGE LOADING ==============

def get_np_img_from_url(url):
    """Fetch an image from a URL as a NumPy array."""
    try:
        img_data = requests.get(url, timeout=5).content
        return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch image: {e}")
        return None


def load_image(req):
    """Load an image from a request containing a base64 string or URL."""
    if "img_b64" in req:
        logging.info("[PROCESSING] Image identified as base64 string.")
        img_data = base64.b64decode(req["img_b64"])
        return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    return get_np_img_from_url(req.get("imageURL", ""))