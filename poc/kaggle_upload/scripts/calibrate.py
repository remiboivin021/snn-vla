"""
Calibration Script for Synobotix VLA PoC

Calibrates the Anchor-Based Quantizer on a dataset.
"""

import argparse
import os
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VLAModel, AnchorQuantizer
from src.data import get_calibration_loader
from src.utils import MemoryProfiler, get_tensor_memory


def calibrate_model(args):
    """
    Calibrate the quantizer for the VLA model.
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = VLAModel(
        config_path=args.model_config,
        quantizer_config_path=args.quant_config
    ).to(device)
    
    if not model.use_quantization:
        print("Quantization is disabled in configuration. Exiting.")
        return
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load calibration data
    print("\nLoading calibration data...")
    calib_loader = get_calibration_loader(
        batch_size=args.batch_size,
        dim=model.dim,
        num_samples=args.num_samples
    )
    
    # Collect samples for calibration
    print("Collecting calibration samples...")
    all_samples = []
    
    for batch_idx, batch in enumerate(calib_loader):
        if batch_idx * args.batch_size >= args.num_samples:
            break
        
        all_samples.append(batch)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Collected {len(all_samples) * args.batch_size} samples...")
    
    # Stack all samples
    all_samples_tensor = torch.cat(all_samples, dim=0).to(device)
    print(f"Collected {all_samples_tensor.shape[0]} samples of shape {all_samples_tensor.shape[1:]}")
    
    # Profile memory before calibration
    print("\nMemory profile before calibration:")
    with MemoryProfiler() as profiler:
        profiler.start()
        # Just hold the samples in memory
        _ = all_samples_tensor.clone()
        profiler.update()
        stats = profiler.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v:.2f} MB")
    
    # Calibrate quantizer
    print("\nCalibrating quantizer...")
    with MemoryProfiler() as profiler:
        profiler.start()
        
        model.quantizer.calibrate(all_samples_tensor, num_samples=args.num_samples)
        
        profiler.update()
        stats = profiler.get_stats()
        print(f"Calibration memory: {stats['cpu_delta']:.2f} MB")
    
    # Test quantization
    print("\nTesting quantization...")
    test_sample = all_samples_tensor[:args.test_size]
    
    # Quantize test sample
    quantized, metadata = model.quantizer.quantize(test_sample)
    
    # Calculate memory savings
    original_memory = get_tensor_memory(test_sample)
    quantized_memory = get_tensor_memory(quantized)
    savings = original_memory - quantized_memory
    savings_percent = (savings / original_memory) * 100
    
    print(f"Original memory: {original_memory:.2f} MB")
    print(f"Quantized memory: {quantized_memory:.2f} MB")
    print(f"Memory savings: {savings:.2f} MB ({savings_percent:.2f}%)")
    
    # Calculate reconstruction error
    dequantized = model.quantizer.dequantize(quantized, metadata)
    mse = torch.mean((test_sample - dequantized) ** 2).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Save calibration
    print("\nSaving calibration...")
    calibration_path = os.path.join(args.output_dir, 'calibration.pt')
    
    state = {
        'quantizer_state': {
            'outlier_mask': model.quantizer.outlier_mask,
            'scales': model.quantizer.scales,
            'zeros': model.quantizer.zeros
        },
        'config': model.quant_config,
        'stats': {
            'num_outliers': model.quantizer.outlier_mask.sum().item(),
            'outlier_ratio': model.quantizer.outlier_mask.float().mean().item(),
            'memory_savings_percent': savings_percent,
            'reconstruction_mse': mse
        }
    }
    
    torch.save(state, calibration_path)
    print(f"Calibration saved to {calibration_path}")
    
    # Print summary
    print("\nCalibration Summary:")
    print(f"  Total dimensions: {model.dim}")
    print(f"  Number of outliers: {state['stats']['num_outliers']}")
    print(f"  Outlier ratio: {state['stats']['outlier_ratio']:.4f}")
    print(f"  Memory savings: {state['stats']['memory_savings_percent']:.2f}%")
    print(f"  Reconstruction MSE: {state['stats']['reconstruction_mse']:.6f}")
    
    # Save to text file
    with open(os.path.join(args.output_dir, 'calibration_summary.txt'), 'w') as f:
        f.write("Calibration Summary\n")
        f.write("=" * 50 + "\n")
        for key, value in state['stats'].items():
            f.write(f"{key}: {value}\n")
        f.write("\nQuantizer Configuration:\n")
        f.write(yaml.dump(state['config'], default_flow_style=False))


def analyze_outliers(args):
    """
    Analyze outlier channels identified during calibration.
    """
    # Load calibration
    calibration_path = args.calibration_path
    if not os.path.exists(calibration_path):
        print(f"Calibration file not found: {calibration_path}")
        return
    
    print(f"Loading calibration from {calibration_path}...")
    state = torch.load(calibration_path, map_location='cpu')
    
    outlier_mask = state['quantizer_state']['outlier_mask']
    scales = state['quantizer_state']['scales']
    zeros = state['quantizer_state']['zeros']
    
    print(f"\nOutlier Analysis:")
    print(f"  Total dimensions: {len(outlier_mask)}")
    print(f"  Number of outliers: {outlier_mask.sum().item()}")
    print(f"  Outlier ratio: {outlier_mask.float().mean().item() * 100:.2f}%")
    
    # Find outlier indices
    outlier_indices = torch.where(outlier_mask)[0].tolist()
    print(f"\nOutlier channel indices: {outlier_indices[:20]}...")
    
    # Analyze scales
    if scales is not None:
        non_outlier_scales = scales[~outlier_mask]
        outlier_scales = scales[outlier_mask]
        
        print(f"\nScale Statistics:")
        print(f"  Non-outliers - mean: {non_outlier_scales.mean():.6f}, std: {non_outlier_scales.std():.6f}")
        print(f"  Outliers - mean: {outlier_scales.mean():.6f}, std: {outlier_scales.std():.6f}")
    
    # Visualize (simple histogram)
    print("\nScale Distribution (histogram):")
    if scales is not None:
        hist, bin_edges = torch.histogram(scales, bins=20)
        for i in range(len(hist)):
            if hist[i] > 0:
                print(f"  [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}): {hist[i].item()}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate Synobotix VLA Quantizer')
    
    # Model arguments
    parser.add_argument('--model_config', type=str, default='config/model.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--quant_config', type=str, default='config/quantization.yaml',
                       help='Path to quantization configuration file')
    
    # Calibration arguments
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for calibration')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples for calibration')
    parser.add_argument('--test_size', type=int, default=1000,
                       help='Number of test samples')
    parser.add_argument('--output_dir', type=str, default='calibration',
                       help='Output directory')
    
    # Analysis
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing calibration')
    parser.add_argument('--calibration_path', type=str, default='calibration/calibration.pt',
                       help='Path to calibration file')
    
    # Hardware
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_outliers(args)
    else:
        calibrate_model(args)


if __name__ == '__main__':
    main()
