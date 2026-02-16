#!/usr/bin/env python3
"""
Cleanup FiftyOne dataset from problematic samples
"""

import fiftyone as fo
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import os
import sys

def check_and_clean_dataset(dataset_name, dry_run=False):
    """
    Validation and cleanup of a FiftyOne dataset
    """
    
    print(f"📖 Loading FiftyOne dataset: {dataset_name}")
    dataset = fo.load_dataset(dataset_name)
    
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    problems = []
    
    print("\n🔍 Checking samples for problems...")
    
    for sample in tqdm(dataset, desc="Checking"):
        sample_problems = []
        
        # 1. Check file existence
        if not os.path.exists(sample.filepath):
            sample_problems.append("File not found")
        else:
            # 2. Image validation
            try:
                img = Image.open(sample.filepath)
                arr = np.array(img)
                
                # Check for NaN/Inf
                if np.isnan(arr).any():
                    sample_problems.append("Image contains NaN")
                if np.isinf(arr).any():
                    sample_problems.append("Image contains Inf")
                
                # Check dimensions
                w, h = img.size
                if w < 10 or h < 10:
                    sample_problems.append(f"Too small: {w}x{h}")
                
                # Check for empty image
                if arr.size == 0:
                    sample_problems.append("Empty image")
                    
            except Exception as e:
                sample_problems.append(f"Cannot read image: {e}")
        
        # 3. Detections check (if applicable)
        if hasattr(sample, 'detections') and sample.detections is not None:
            if len(sample.detections.detections) == 0:
                sample_problems.append("No detections")
            else:
                for i, det in enumerate(sample.detections.detections):
                    bbox = det.bounding_box
                    
                    if bbox is None:
                        sample_problems.append(f"Detection {i}: No bbox")
                        continue
                    
                    x, y, w, h = bbox
                    
                    # Coordinate validity check
                    if w <= 0 or h <= 0:
                        sample_problems.append(f"Detection {i}: Invalid size (w={w}, h={h})")
                    
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        sample_problems.append(f"Detection {i}: bbox out of bounds (x={x}, y={y})")
                    
                    if w > 1 or h > 1:
                        sample_problems.append(f"Detection {i}: bbox too large (w={w}, h={h})")
                    
                    # Check for NaN/Inf in bbox
                    if any(np.isnan([x, y, w, h])):
                        sample_problems.append(f"Detection {i}: Contains NaN")
                    if any(np.isinf([x, y, w, h])):
                        sample_problems.append(f"Detection {i}: Contains Inf")
                    
                    # Check for extremely small bboxes
                    if w < 0.001 or h < 0.001:
                        sample_problems.append(f"Detection {i}: bbox too small (w={w}, h={h})")
        else:
            sample_problems.append("No detections field")
        
        if sample_problems:
            problems.append({
                'id': sample.id,
                'filepath': sample.filepath,
                'issues': sample_problems
            })
    
    # Results Summary
    print(f"\n{'='*80}")
    print(f"✅ Total samples checked: {total_samples}")
    print(f"❌ Samples with issues: {len(problems)}")
    print(f"✅ Clean samples: {total_samples - len(problems)}")
    print(f"{'='*80}\n")
    
    if len(problems) == 0:
        print("✅ Dataset is clean!")
        return
    
    # Show statistics by problem type
    issue_types = {}
    for p in problems:
        for issue in p['issues']:
            # Extract issue type
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    print("📊 Problem types:")
    for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {issue_type}: {count}")
    
    # Show examples
    print(f"\n🚨 First 10 problematic samples:")
    for i, p in enumerate(problems[:10], 1):
        print(f"\n{i}. {p['filepath']}")
        for issue in p['issues']:
            print(f"   ❌ {issue}")
    
    if len(problems) > 10:
        print(f"\n... and {len(problems) - 10} more")
    
    # Cleanup Process
    if not dry_run:
        print(f"\n⚠️  Removing {len(problems)} problematic samples from dataset...")
        
        response = input(f"This will DELETE {len(problems)} samples from '{dataset_name}'. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        
        # Create a view with problematic samples
        problem_ids = [p['id'] for p in problems]
        problem_view = dataset.select(problem_ids)
        
        # Delete
        print("Deleting...")
        dataset.delete_samples(problem_view)
        
        print(f"✅ Removed {len(problems)} samples")
        print(f"✅ Remaining samples: {len(dataset)}")
        
        # Save Report
        report_path = 'fiftyone_cleanup_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"FiftyOne Dataset Cleanup Report\n")
            f.write(f"{'='*80}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Original samples: {total_samples}\n")
            f.write(f"Removed samples: {len(problems)}\n")
            f.write(f"Remaining samples: {len(dataset)}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write("Problem types:\n")
            for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {issue_type}: {count}\n")
            
            f.write(f"\nRemoved samples details:\n")
            for p in problems:
                f.write(f"\n{p['filepath']}\n")
                for issue in p['issues']:
                    f.write(f"  {issue}\n")
        
        print(f"\n📄 Report saved to: {report_path}")
    else:
        print(f"\n🔍 DRY RUN - no changes made to dataset")

def main():
    parser = argparse.ArgumentParser(
        description='Clean FiftyOne dataset from problematic samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - see what would be removed
  python clean_fiftyone_dataset.py --dataset segmentation_merged_v0.1_full --dry-run
  
  # Clean dataset (actually removes bad samples)
  python clean_fiftyone_dataset.py --dataset segmentation_merged_v0.1_full
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='FiftyOne dataset name')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be done without deleting')
    
    args = parser.parse_args()
    
    try:
        check_and_clean_dataset(args.dataset, args.dry_run)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()