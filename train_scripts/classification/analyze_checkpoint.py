#!/usr/bin/env python3
"""
Analyze PyTorch Lightning checkpoint to see what takes up space.

Usage:
    python analyze_checkpoint.py --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import os
import torch
import sys
from collections import OrderedDict


def get_tensor_size_mb(tensor):
    """Calculate size of a tensor in MB."""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement() / (1024 * 1024)
    return 0


def analyze_state_dict(state_dict, name="state_dict"):
    """Analyze a state dictionary and return size breakdown."""
    total_size = 0
    component_sizes = {}
    
    for key, value in state_dict.items():
        size = get_tensor_size_mb(value)
        total_size += size
        
        # Group by component (model, optimizer, etc.)
        component = key.split('.')[0] if '.' in key else key
        component_sizes[component] = component_sizes.get(component, 0) + size
    
    return total_size, component_sizes


def analyze_checkpoint(checkpoint_path: str, show_details: bool = False):
    """
    Analyze checkpoint and print detailed size breakdown.
    
    Args:
        checkpoint_path: Path to checkpoint file
        show_details: Whether to show detailed component breakdown
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"{'='*70}\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Analyze top-level keys
    print("Top-level keys:")
    print("-" * 70)
    
    total_analyzed = 0
    key_sizes = {}
    
    for key in checkpoint.keys():
        value = checkpoint[key]
        
        if isinstance(value, OrderedDict) or isinstance(value, dict):
            size, components = analyze_state_dict(value, key)
            key_sizes[key] = (size, components)
            total_analyzed += size
            print(f"  {key:30s} {size:10.2f} MB  (dict with {len(value)} items)")
            
            if show_details and size > 0.1:  # Show details for components > 0.1 MB
                for comp_name, comp_size in sorted(components.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    └─ {comp_name:26s} {comp_size:10.2f} MB")
        
        elif isinstance(value, torch.Tensor):
            size = get_tensor_size_mb(value)
            key_sizes[key] = (size, {})
            total_analyzed += size
            print(f"  {key:30s} {size:10.2f} MB  (tensor: {list(value.shape)})")
        
        else:
            # For other types (strings, numbers, etc.)
            size = 0
            key_sizes[key] = (size, {})
            print(f"  {key:30s} {'<0.01':>10s} MB  ({type(value).__name__})")
    
    print("-" * 70)
    print(f"  {'Total analyzed:':30s} {total_analyzed:10.2f} MB")
    print(f"  {'Overhead/metadata:':30s} {file_size_mb - total_analyzed:10.2f} MB")
    print()
    
    # Estimate size reduction if using save_weights_only
    if 'state_dict' in checkpoint:
        state_dict_size = key_sizes.get('state_dict', (0, {}))[0]
        hyper_params_size = key_sizes.get('hyper_parameters', (0, {}))[0]
        estimated_lightweight_size = state_dict_size + hyper_params_size + (file_size_mb - total_analyzed)
        
        potential_savings = file_size_mb - estimated_lightweight_size
        savings_percent = (potential_savings / file_size_mb) * 100
        
        print("Potential Optimization:")
        print("-" * 70)
        print(f"  Using save_weights_only=True would save approximately:")
        print(f"  {potential_savings:.2f} MB ({savings_percent:.1f}% reduction)")
        print(f"  New estimated size: {estimated_lightweight_size:.2f} MB")
        print()
    
    # Show what takes the most space
    print("Components taking the most space:")
    print("-" * 70)
    sorted_keys = sorted(key_sizes.items(), key=lambda x: x[1][0], reverse=True)
    for key, (size, _) in sorted_keys[:5]:
        if size > 0:
            percent = (size / file_size_mb) * 100
            print(f"  {key:30s} {size:10.2f} MB  ({percent:.1f}%)")
    print()
    
    # Additional checkpoint info
    print("Checkpoint Metadata:")
    print("-" * 70)
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"  Global Step: {checkpoint['global_step']}")
    if 'pytorch-lightning_version' in checkpoint:
        print(f"  PyTorch Lightning Version: {checkpoint['pytorch-lightning_version']}")
    
    # Count parameters in state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Model Layers: {len(state_dict)}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch Lightning checkpoint size",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show detailed component breakdown"
    )
    
    args = parser.parse_args()
    
    analyze_checkpoint(args.checkpoint, show_details=args.details)


if __name__ == "__main__":
    main()
