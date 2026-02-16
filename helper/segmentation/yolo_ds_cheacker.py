import os
import numpy as np
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--labels", type=str, required=True, help="Path to YOLO label folder")
parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
parser.add_argument("--min_pixels", type=int, default=3, help="Min bbox size in pixels")
parser.add_argument("--min_area", type=int, default=25, help="Min bbox area in pixels")
args = parser.parse_args()

LABEL_DIR = args.labels
IMG_SIZE = args.imgsz
MIN_PIXELS = args.min_pixels
MIN_AREA = args.min_area

issues = defaultdict(list)

def polygon_area(coords):
    """Shoelace formula"""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

from tqdm import tqdm

for file in tqdm(os.listdir(LABEL_DIR)):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, file)

    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        issues["empty_file"].append(file)
        continue

    for i, line in enumerate(lines):
        try:
            nums = list(map(float, line.split()))
        except:
            issues["parse_error"].append(file)
            break

        if len(nums) < 7:  # class + min 3 points
            issues["too_few_points"].append(file)
            break

        coords = np.array(nums[1:]).reshape(-1, 2)

        # NaN / Inf check
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            issues["nan_or_inf"].append(file)
            break

        # Range check
        if np.any(coords < 0) or np.any(coords > 1):
            issues["out_of_range"].append(file)
            break

        # BBox size check
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        w = (x_max - x_min) * IMG_SIZE
        h = (y_max - y_min) * IMG_SIZE
        area = w * h

        if w < MIN_PIXELS or h < MIN_PIXELS:
            issues["tiny_bbox"].append(file)

        if area < MIN_AREA:
            issues["tiny_area"].append(file)

        # Polygon area check
        poly_area = polygon_area(coords) * (IMG_SIZE ** 2)
        if poly_area < 1:
            issues["degenerate_polygon"].append(file)


# ===== REPORT =====

print("\n===== DATASET CHECK REPORT =====\n")

total_problem_files = set()
for k, v in issues.items():
    unique_files = set(v)
    total_problem_files.update(unique_files)
    print(f"{k:20s}: {len(unique_files)} files")

print("\nTotal problematic files:", len(total_problem_files))

if total_problem_files:
    print("\nExample problematic files:")
    for f in list(total_problem_files)[:20]:
        print(" -", f)

print("\nDone.")
