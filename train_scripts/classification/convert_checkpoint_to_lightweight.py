#!/usr/bin/env python3
"""
Utility to convert full PyTorch Lightning checkpoints to lightweight versions.

This script removes optimizer states and other training metadata, keeping only
the model weights. This can reduce checkpoint size by 3-4x.

Usage:
    python convert_checkpoint_to_lightweight.py --input checkpoint.ckpt --output checkpoint_weights_only.ckpt
    
    # Batch convert all checkpoints in a directory
    python convert_checkpoint_to_lightweight.py --input_dir /path/to/checkpoints/ --output_dir /path/to/output/
"""

import argparse
import os
import torch
from pathlib import Path


def convert_checkpoint(input_path: str, output_path: str, verbose: bool = True):
    """
    Convert a full checkpoint to weights-only format.
    
    Args:
        input_path: Path to input checkpoint
        output_path: Path to save lightweight checkpoint
        verbose: Whether to print size information
    """
    if verbose:
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"Loading checkpoint from: {input_path}")
        print(f"Original size: {input_size_mb:.2f} MB")
    
    # Load checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Extract only the essential parts
    lightweight_checkpoint = {
        'state_dict': checkpoint['state_dict'],
        'hyper_parameters': checkpoint.get('hyper_parameters', {}),
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'pytorch-lightning_version': checkpoint.get('pytorch-lightning_version', ''),
    }
    
    # Save lightweight checkpoint
    torch.save(lightweight_checkpoint, output_path)
    
    if verbose:
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        reduction_percent = ((input_size_mb - output_size_mb) / input_size_mb) * 100
        print(f"Saved to: {output_path}")
        print(f"New size: {output_size_mb:.2f} MB")
        print(f"Reduction: {reduction_percent:.1f}% ({input_size_mb - output_size_mb:.2f} MB saved)")
        print()


def convert_directory(input_dir: str, output_dir: str, pattern: str = "*.ckpt"):
    """
    Convert all checkpoints in a directory.
    
    Args:
        input_dir: Input directory containing checkpoints
        output_dir: Output directory for lightweight checkpoints
        pattern: Glob pattern to match checkpoint files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_files = list(input_path.glob(pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found matching pattern '{pattern}' in {input_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint(s) to convert\n")
    
    total_saved = 0
    for ckpt_file in sorted(checkpoint_files):
        output_file = output_path / ckpt_file.name
        
        input_size_mb = os.path.getsize(ckpt_file) / (1024 * 1024)
        convert_checkpoint(str(ckpt_file), str(output_file), verbose=True)
        output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        total_saved += (input_size_mb - output_size_mb)
    
    print(f"Conversion complete!")
    print(f"Total space saved: {total_saved:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Lightning checkpoints to lightweight format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input checkpoint file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output checkpoint file path"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing checkpoints (for batch conversion)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for lightweight checkpoints (for batch conversion)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.ckpt",
        help="Glob pattern for checkpoint files (default: *.ckpt)"
    )
    
    args = parser.parse_args()
    
    # Single file conversion
    if args.input and args.output:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return
        convert_checkpoint(args.input, args.output)
    
    # Batch directory conversion
    elif args.input_dir and args.output_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            return
        convert_directory(args.input_dir, args.output_dir, args.pattern)
    
    else:
        parser.print_help()
        print("\nError: You must specify either:")
        print("  1. --input and --output for single file conversion, or")
        print("  2. --input_dir and --output_dir for batch conversion")


if __name__ == "__main__":
    main()
