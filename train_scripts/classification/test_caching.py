#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify dataset caching functionality.

This script creates a simple test to demonstrate that caching works correctly.
"""

import os
import sys
import time
import tempfile
import shutil

# Add project root to path
CURRENT_FOLDER_PATH = os.path.abspath(__file__)
DELIMITER = 'fish-identification'
pos = CURRENT_FOLDER_PATH.find(DELIMITER)
if pos != -1:
    sys.path.insert(1, CURRENT_FOLDER_PATH[:pos + len(DELIMITER)])

from module.classification_package.src.datamodule import ImageEmbeddingDataModule


def test_caching_speed():
    """
    Test caching functionality and measure speed improvements.
    
    NOTE: This test requires a valid FiftyOne dataset to run.
    Modify the dataset_name below to match your setup.
    """
    
    # Configuration - MODIFY THESE FOR YOUR SETUP
    dataset_name = "classification_v0.10_validation"  # Change to your dataset
    train_tag = "train"
    val_tag = "val"
    
    print("=" * 80)
    print("⚠️  IMPORTANT: Please verify these settings before running:")
    print("=" * 80)
    print(f"Dataset name: {dataset_name}")
    print(f"Train tag: {train_tag}")
    print(f"Val tag: {val_tag}")
    print()
    response = input("Continue with these settings? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Test cancelled.")
        return False
    print()
    
    # Create temporary cache directory for testing
    temp_cache_dir = tempfile.mkdtemp(prefix="test_cache_")
    
    try:
        print("=" * 80)
        print("Dataset Caching Test")
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Train tag: {train_tag}")
        print(f"Val tag: {val_tag}")
        print(f"Temp cache dir: {temp_cache_dir}")
        print()
        
        # Test 1: First run (no cache)
        print("-" * 80)
        print("TEST 1: First run (should process dataset from scratch)")
        print("-" * 80)
        
        datamodule1 = ImageEmbeddingDataModule(
            dataset_name=dataset_name,
            batch_size=32,
            classes_per_batch=24,
            samples_per_class=2,
            image_size=224,
            num_workers=4,
            train_tag=train_tag,
            val_tag=val_tag,
            cache_dir=temp_cache_dir,
            use_cache=True,
        )
        
        start_time = time.time()
        datamodule1.setup()
        first_run_time = time.time() - start_time
        
        print(f"✓ First run completed in {first_run_time:.2f} seconds")
        print(f"  Train classes: {datamodule1.num_classes}")
        print(f"  Train samples: {len(datamodule1.train_dataset)}")
        if datamodule1.val_dataset:
            print(f"  Val samples: {len(datamodule1.val_dataset)}")
        print()
        
        # Test 2: Second run (with cache)
        print("-" * 80)
        print("TEST 2: Second run (should load from cache)")
        print("-" * 80)
        
        datamodule2 = ImageEmbeddingDataModule(
            dataset_name=dataset_name,
            batch_size=32,
            classes_per_batch=24,
            samples_per_class=2,
            image_size=224,
            num_workers=4,
            train_tag=train_tag,
            val_tag=val_tag,
            cache_dir=temp_cache_dir,
            use_cache=True,
        )
        
        start_time = time.time()
        datamodule2.setup()
        cached_run_time = time.time() - start_time
        
        print(f"✓ Cached run completed in {cached_run_time:.2f} seconds")
        print(f"  Train classes: {datamodule2.num_classes}")
        print(f"  Train samples: {len(datamodule2.train_dataset)}")
        if datamodule2.val_dataset:
            print(f"  Val samples: {len(datamodule2.val_dataset)}")
        print()
        
        # Test 3: Run with cache disabled
        print("-" * 80)
        print("TEST 3: Run with cache disabled (should process from scratch)")
        print("-" * 80)
        
        datamodule3 = ImageEmbeddingDataModule(
            dataset_name=dataset_name,
            batch_size=32,
            classes_per_batch=24,
            samples_per_class=2,
            image_size=224,
            num_workers=4,
            train_tag=train_tag,
            val_tag=val_tag,
            cache_dir=temp_cache_dir,
            use_cache=False,  # Disabled
        )
        
        start_time = time.time()
        datamodule3.setup()
        no_cache_run_time = time.time() - start_time
        
        print(f"✓ No-cache run completed in {no_cache_run_time:.2f} seconds")
        print()
        
        # Summary
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"First run (create cache):  {first_run_time:.2f} seconds")
        print(f"Cached run (load cache):   {cached_run_time:.2f} seconds")
        print(f"No-cache run:              {no_cache_run_time:.2f} seconds")
        print()
        
        speedup = first_run_time / cached_run_time if cached_run_time > 0 else 0
        print(f"Speedup with cache: {speedup:.1f}x faster")
        print()
        
        # Validation
        if cached_run_time < first_run_time * 0.5:
            print("✓ SUCCESS: Caching provides significant speedup!")
        else:
            print("⚠ WARNING: Cache speedup is less than expected")
        
        if datamodule1.num_classes == datamodule2.num_classes == datamodule3.num_classes:
            print("✓ SUCCESS: All runs produced consistent results")
        else:
            print("✗ ERROR: Results are inconsistent across runs")
        
        print()
        
    except Exception as e:
        print(f"✗ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(temp_cache_dir):
            shutil.rmtree(temp_cache_dir)
            print(f"Cleaned up temp cache dir: {temp_cache_dir}")
    
    return True


if __name__ == "__main__":
    print()
    print("NOTE: This test requires a valid FiftyOne dataset.")
    print("Please edit the script to set the correct dataset_name.")
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="Test dataset caching")
    parser.add_argument("--run", action="store_true", help="Run the test")
    args = parser.parse_args()
    
    if args.run:
        success = test_caching_speed()
        sys.exit(0 if success else 1)
    else:
        print("Use --run flag to execute the test")
        print("Example: python test_caching.py --run")
