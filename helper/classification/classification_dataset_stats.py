
import fiftyone as fo
from collections import Counter
import sys

def main():
    dataset_name = "classification_v0.10"
    
    print(f"Loading dataset: {dataset_name}")
    if dataset_name not in fo.list_datasets():
        print(f"Error: Dataset '{dataset_name}' not found in FiftyOne.")
        print("Available datasets:", fo.list_datasets())
        sys.exit(1)

    dataset = fo.load_dataset(dataset_name)
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    print("Computing statistics...")
    label_counts = Counter()
    
    # Iterate over samples to count labels
    # Based on classification_dataset_creator.py, the label is in sample.polyline.label
    for sample in dataset.iter_samples(progress=True):
        if "polyline" in sample:
            poly = sample.polyline
            if poly is None:
                continue
            
            # Check if polyline is a single object or list (FiftyOne field types can vary)
            if isinstance(poly, fo.Polyline):
                label = poly.label
                label_counts[label] += 1
            elif isinstance(poly, list):
                # If it's a list (Polylines), iterate
                for p in poly:
                    label_counts[p.label] += 1
            else:
                # Handle unexpected type if possible or skip
                pass

    print("\n" + "="*50)
    print(f"Statistics for '{dataset_name}'")
    print("="*50)
    print(f"Total Unique Classes: {len(label_counts)}")
    print(f"Total Annotated Items: {sum(label_counts.values())}")
    print("-" * 50)
    print(f"{'Class Name':<35} | {'Count':<10}")
    print("-" * 50)
    
    for label, count in label_counts.most_common():
        print(f"{label:<35} | {count:<10}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()
