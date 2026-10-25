#!/usr/bin/env python3
"""
Main entry point for Synobotix VLA PoC

Provides a simple command-line interface for common operations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: str):
    """Run a shell command and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {command}")
    print('=' * 60)
    
    result = subprocess.run(command, shell=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Synobotix VLA PoC - Main Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test              # Run all unit tests
  python main.py train             # Train the model
  python main.py calibrate         # Calibrate the quantizer
  python main.py evaluate          # Evaluate the model
  python main.py test gqa          # Run GQA tests only
  python main.py info              # Show project info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run unit tests')
    test_parser.add_argument('module', nargs='?', default='all',
                           choices=['all', 'gqa', 'quantization', 'eviction'],
                           help='Test module to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', default='config/model.yaml',
                             help='Model configuration file')
    train_parser.add_argument('--quant-config', default='config/quantization.yaml',
                             help='Quantization configuration file')
    train_parser.add_argument('--epochs', type=int, default=10,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size')
    train_parser.add_argument('--output', default='outputs/train',
                             help='Output directory')
    
    # Calibrate command
    calib_parser = subparsers.add_parser('calibrate', help='Calibrate quantizer')
    calib_parser.add_argument('--config', default='config/model.yaml',
                             help='Model configuration file')
    calib_parser.add_argument('--quant-config', default='config/quantization.yaml',
                             help='Quantization configuration file')
    calib_parser.add_argument('--samples', type=int, default=10000,
                             help='Number of calibration samples')
    calib_parser.add_argument('--output', default='calibration',
                             help='Output directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('type', choices=['perplexity', 'memory', 'robotic', 'all'],
                           help='Type of evaluation')
    eval_parser.add_argument('--model', required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--output', default='outputs/eval',
                           help='Output directory')
    
    # Info command
    subparsers.add_parser('info', help='Show project information')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'test':
        if args.module == 'all':
            cmd = f"python -m tests"
        else:
            cmd = f"python -m pytest tests/test_{args.module}.py -v"
        
        return_code = run_command(cmd)
        sys.exit(return_code)
    
    elif args.command == 'train':
        cmd = (f"python scripts/train.py "
               f"--model_config {args.config} "
               f"--quant_config {args.quant_config} "
               f"--num_epochs {args.epochs} "
               f"--batch_size {args.batch_size} "
               f"--output_dir {args.output} "
               f"--calibrate")
        
        return_code = run_command(cmd)
        sys.exit(return_code)
    
    elif args.command == 'calibrate':
        cmd = (f"python scripts/calibrate.py "
               f"--model_config {args.config} "
               f"--quant_config {args.quant_config} "
               f"--num_samples {args.samples} "
               f"--output_dir {args.output}")
        
        return_code = run_command(cmd)
        sys.exit(return_code)
    
    elif args.command == 'evaluate':
        cmd = (f"python scripts/evaluate.py "
               f"--model_path {args.model} "
               f"--eval_type {args.type} "
               f"--output_dir {args.output}")
        
        return_code = run_command(cmd)
        sys.exit(return_code)
    
    elif args.command == 'info':
        print("\n" + "=" * 60)
        print("Synobotix VLA PoC - Project Information")
        print("=" * 60)
        
        print("\n📁 Project Structure:")
        result = subprocess.run("tree -L 3 -I '__pycache__|.git|venv' . 2>/dev/null || find . -maxdepth 3 -type f | grep -v __pycache__ | grep -v '.git' | sort", 
                       shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        print(result.stdout)
        
        print("\n📊 Implementation Status:")
        print("  ✅ Pivot 1: GQA Architecture (src/models/gqa.py)")
        print("  ✅ Pivot 2: Anchor-Based Quantization (src/models/quantizer.py)")
        print("  ✅ Pivot 3: Asymmetric Eviction (src/models/kv_cache.py)")
        print("  ✅ Axe 4: Gated DeltaNet (src/models/delta_net.py)")
        print("\n🎯 Objectives:")
        print("  • 8x context increase (4,096 → 32,768 tokens)")
        print("  • Memory ≤ 330 Mio RAM")
        print("  • < 1 point perplexity degradation")
        print("  • Stable task success rate on 5-min sequences")
        
        print("\n🚀 Quick Commands:")
        print("  python main.py test              # Run all tests")
        print("  python main.py train             # Train the model")
        print("  python main.py calibrate         # Calibrate quantizer")
        print("  python main.py evaluate all --model outputs/train/model_best.pt  # Full evaluation")
        
        print("\n📚 Documentation:")
        print("  • Full documentation in README.md")
        print("  • Configuration in config/*.yaml")
        print("  • Source code in src/")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
