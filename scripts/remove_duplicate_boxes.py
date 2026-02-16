#!/usr/bin/env python3
"""
Find and remove duplicate bounding boxes in a FiftyOne dataset
"""

import fiftyone as fo
from tqdm import tqdm
import numpy as np
import argparse
import sys

def find_and_remove_duplicates(dataset_name, dry_run=False, tolerance=1e-6):
    """
    Find and remove duplicate bounding boxes
    
    Args:
        dataset_name: Name of the FiftyOne dataset
        dry_run: Only show statistics without modifying the database
        tolerance: Coordinate comparison tolerance
    """
    
    print(f"📖 Loading dataset: {dataset_name}")
    dataset = fo.load_dataset(dataset_name)
    
    print(f"Total samples: {len(dataset)}")
    
    # Detect field with detections
    sample = dataset.first()
    detection_field = None
    
    for field in sample.field_names:
        field_value = getattr(sample, field, None)
        if hasattr(field_value, 'detections'):
            detection_field = field
            print(f"✓ Detection field: {field}")
            break
    
    if not detection_field:
        print("❌ No detection field found!")
        return
    
    # Statistics
    total_detections = 0
    total_duplicates = 0
    samples_with_duplicates = 0
    samples_modified = []
    
    print(f"\n🔍 Checking for duplicate bounding boxes...")
    
    for sample in tqdm(dataset, desc="Processing"):
        detections = getattr(sample, detection_field)
        
        if not detections or not hasattr(detections, 'detections'):
            continue
        
        dets = detections.detections
        if len(dets) <= 1:
            total_detections += len(dets)
            continue
        
        # Check for duplicates
        unique_boxes = []
        duplicate_indices = []
        
        for i, det in enumerate(dets):
            bbox = det.bounding_box
            
            # Check if an identical bbox already exists
            is_duplicate = False
            for unique_bbox in unique_boxes:
                # Comparison with tolerance
                if all(abs(bbox[j] - unique_bbox[j]) < tolerance for j in range(4)):
                    is_duplicate = True
                    duplicate_indices.append(i)
                    total_duplicates += 1
                    break
            
            if not is_duplicate:
                unique_boxes.append(bbox)
        
        total_detections += len(dets)
        
        # If duplicates found in this sample
        if duplicate_indices:
            samples_with_duplicates += 1
            samples_modified.append({
                'sample_id': sample.id,
                'filepath': sample.filepath,
                'total_dets': len(dets),
                'duplicates': len(duplicate_indices),
                'unique': len(unique_boxes)
            })
            
            # Remove duplicates
            if not dry_run:
                # Create list of unique detections
                unique_dets = [det for i, det in enumerate(dets) if i not in duplicate_indices]
                
                # Update sample
                new_detections = fo.Detections(detections=unique_dets)
                sample[detection_field] = new_detections
                sample.save()
    
    # Results Summary
    print(f"\n{'='*80}")
    print(f"📊 RESULTS:")
    print(f"{'='*80}")
    print(f"Total detections: {total_detections}")
    print(f"Duplicate detections: {total_duplicates} ({total_duplicates/total_detections*100:.2f}%)")
    print(f"Samples with duplicates: {samples_with_duplicates}")
    print(f"{'='*80}\n")
    
    if samples_with_duplicates > 0:
        print(f"🚨 Samples with duplicates (first 20):")
        for i, info in enumerate(samples_modified[:20], 1):
            print(f"{i}. {info['filepath']}")
            print(f"   Total: {info['total_dets']}, Unique: {info['unique']}, Duplicates: {info['duplicates']}")
        
        if len(samples_modified) > 20:
            print(f"... and {len(samples_modified) - 20} more samples")
        
        print()
        
        if dry_run:
            print("🔍 DRY RUN - no changes made")
            print(f"\nTo apply changes, run without --dry-run")
        else:
            print(f"✅ Removed {total_duplicates} duplicate detections!")
            print(f"✅ Modified {samples_with_duplicates} samples")
            
            # Save report
            report_path = 'duplicate_removal_report.txt'
            with open(report_path, 'w') as f:
                f.write(f"Duplicate Bounding Boxes Removal Report\n")
                f.write(f"{'='*80}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Total detections: {total_detections}\n")
                f.write(f"Duplicate detections removed: {total_duplicates}\n")
                f.write(f"Samples modified: {samples_with_duplicates}\n")
                f.write(f"{'='*80}\n\n")
                
                for info in samples_modified:
                    f.write(f"\n{info['filepath']}\n")
                    f.write(f"  Total: {info['total_dets']}, Unique: {info['unique']}, Duplicates: {info['duplicates']}\n")
            
            print(f"\n📄 Report saved to: {report_path}")
    else:
        print("✅ No duplicate bounding boxes found!")
    
    return total_duplicates

def main():
    parser = argparse.ArgumentParser(
        description='Find and remove duplicate bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - see what would be removed
  python remove_duplicate_boxes.py --dataset segmentation_merged_v0.1_full --dry-run
  
  # Remove duplicates
  python remove_duplicate_boxes.py --dataset segmentation_merged_v0.1_full
  
  # With custom tolerance
  python remove_duplicate_boxes.py --dataset segmentation_merged_v0.1_full --tolerance 0.0001
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='FiftyOne dataset name')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be done')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Tolerance for comparing coordinates (default: 1e-6)')
    
    args = parser.parse_args()
    
    try:
        duplicates = find_and_remove_duplicates(args.dataset, args.dry_run, args.tolerance)
        
        if duplicates > 0:
            print(f"\n⚠️  Found {duplicates} duplicates")
            if args.dry_run:
                print("Run without --dry-run to remove them")
                sys.exit(1)
            else:
                print("✅ Duplicates removed! You may need to export/recreate your YOLO dataset now.")
                sys.exit(0)
        else:
            print("✅ No duplicates found")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()