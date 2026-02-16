#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to clear dataset cache files.

This script helps manage the dataset cache created by the training pipeline.
You can clear all caches or specific ones by dataset name.

Usage:
    # Clear all cache files
    python clear_dataset_cache.py --all
    
    # Clear cache for a specific dataset
    python clear_dataset_cache.py --dataset_name "my_dataset"
    
    # Clear cache from a specific directory
    python clear_dataset_cache.py --cache_dir "/path/to/cache" --all
    
    # List all cache files without deleting
    python clear_dataset_cache.py --list
"""

import argparse
import os
import glob
import json
from pathlib import Path


def get_default_cache_dir():
    """Get the default cache directory."""
    return os.path.expanduser("~/.cache/fish_identification")


def list_cache_files(cache_dir):
    """
    List all cache files in the cache directory.
    
    Args:
        cache_dir: Path to cache directory
        
    Returns:
        List of (file_path, metadata_dict) tuples
    """
    cache_files = []
    pattern = os.path.join(cache_dir, "dataset_cache_*.json")
    
    for cache_file in glob.glob(pattern):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract relevant metadata
            info = {
                'path': cache_file,
                'dataset_name': metadata.get('dataset_name', 'unknown'),
                'train_tag': metadata.get('train_tag', 'None'),
                'val_tag': metadata.get('val_tag', 'None'),
                'train_classes': len(metadata.get('train_records', {})),
                'val_classes': len(metadata.get('val_records', {})),
                'size_mb': os.path.getsize(cache_file) / (1024 * 1024),
            }
            cache_files.append(info)
        except Exception as e:
            print(f"Warning: Could not read {cache_file}: {e}")
    
    return cache_files


def clear_cache(cache_dir, dataset_name=None, clear_all=False):
    """
    Clear cache files.
    
    Args:
        cache_dir: Path to cache directory
        dataset_name: Specific dataset name to clear (optional)
        clear_all: Clear all cache files
        
    Returns:
        Number of files deleted
    """
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return 0
    
    cache_files = list_cache_files(cache_dir)
    
    if not cache_files:
        print(f"No cache files found in: {cache_dir}")
        return 0
    
    deleted_count = 0
    
    for cache_info in cache_files:
        should_delete = False
        
        if clear_all:
            should_delete = True
        elif dataset_name and cache_info['dataset_name'] == dataset_name:
            should_delete = True
        
        if should_delete:
            try:
                os.remove(cache_info['path'])
                print(f"Deleted: {cache_info['path']}")
                print(f"  Dataset: {cache_info['dataset_name']}, "
                      f"Train tag: {cache_info['train_tag']}, "
                      f"Val tag: {cache_info['val_tag']}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {cache_info['path']}: {e}")
    
    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Clear dataset cache files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory path (default: ~/.cache/fish_identification)"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Clear cache only for this specific dataset"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear all cache files"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all cache files without deleting"
    )
    
    args = parser.parse_args()
    
    cache_dir = args.cache_dir if args.cache_dir else get_default_cache_dir()
    
    print(f"Cache directory: {cache_dir}")
    print()
    
    if args.list:
        cache_files = list_cache_files(cache_dir)
        if not cache_files:
            print("No cache files found.")
        else:
            print(f"Found {len(cache_files)} cache file(s):")
            print()
            for i, info in enumerate(cache_files, 1):
                print(f"{i}. {Path(info['path']).name}")
                print(f"   Dataset: {info['dataset_name']}")
                print(f"   Train tag: {info['train_tag']}, Val tag: {info['val_tag']}")
                print(f"   Train classes: {info['train_classes']}, Val classes: {info['val_classes']}")
                print(f"   Size: {info['size_mb']:.2f} MB")
                print()
    else:
        if not args.all and not args.dataset_name:
            print("Error: Please specify --all or --dataset_name")
            parser.print_help()
            return
        
        deleted = clear_cache(cache_dir, args.dataset_name, args.all)
        print()
        print(f"Deleted {deleted} cache file(s).")


if __name__ == "__main__":
    main()
