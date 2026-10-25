"""
Training Script for Synobotix VLA PoC

Trains the VLA model with all 4 R&D axes.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import datetime

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VLAModel
from src.data import get_dataloader, get_calibration_loader
from src.utils import MemoryProfiler, MetricsTracker, compute_perplexity


def train(args):
    """
    Main training function.
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

    # DataParallel splits each batch across GPUs; custom methods (save, calibrate...) stay on `model`
    forward = torch.nn.DataParallel(model) if device.type == 'cuda' and torch.cuda.device_count() > 1 else model
    print(f"GPUs used: {torch.cuda.device_count() if forward is not model else 1}")
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    memory_usage = model.get_memory_usage()
    print(f"Initial memory usage: {memory_usage['total']:.2f} MB")
    
    # Load data
    print("\nLoading data...")
    dataset_kwargs = {}
    if args.dataset_type == 'lerobot':
        dataset_kwargs = {'repo_id': args.repo_id, 'image_keys': args.image_keys,
                          'frame_cache_dir': args.frame_cache_dir}
    train_loader = get_dataloader(
        dataset_type=args.dataset_type,
        **dataset_kwargs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size
    )
    
    val_loader = get_dataloader(
        dataset_type=args.dataset_type,
        **dataset_kwargs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        split='val'
    )
    
    # Calibrate quantizer if needed
    if model.use_quantization and args.calibrate:
        print("\nCalibrating quantizer...")
        calib_loader = get_calibration_loader(
            batch_size=args.calib_batch_size,
            dim=model.dim
        )
        model.calibrate_quantizer(calib_loader, num_samples=args.num_calib_samples)
        print("Quantizer calibration complete.")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_loader)
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ppl = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            text_tokens = batch['text_tokens'].to(device)
            # For PoC, we'll use text tokens only
            # In full implementation, we'd use vision_patches and metadata too
            
            # Forward pass
            logits = forward(text_tokens=text_tokens)
            
            # Compute loss
            # Shift targets for next token prediction
            targets = batch['target_tokens'].to(device)
            
            # Reshape for loss computation
            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            
            # Compute perplexity
            ppl = compute_perplexity(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_ppl += ppl
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{ppl:.2f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            metrics.add('train_loss', loss.item())
            metrics.add('train_ppl', ppl)
            metrics.increment_step()
            
            # Log memory usage
            if step % args.log_interval == 0:
                with torch.no_grad():
                    memory_usage = model.get_memory_usage()
                    progress_bar.write(f"Memory: {memory_usage['total']:.2f} MB")
        
        # Calculate epoch metrics
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_ppl = epoch_ppl / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_ppl = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                text_tokens = batch['text_tokens'].to(device)
                logits = forward(text_tokens=text_tokens)
                
                targets = batch['target_tokens'].to(device)
                loss = criterion(
                    logits.view(-1, logits.shape[-1]),
                    targets.view(-1)
                )
                
                ppl = compute_perplexity(logits, targets)
                val_loss += loss.item()
                val_ppl += ppl
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ppl = val_ppl / len(val_loader)
        
        metrics.add('val_loss', avg_val_loss)
        metrics.add('val_ppl', avg_val_ppl)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train PPL: {avg_train_ppl:.2f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val PPL: {avg_val_ppl:.2f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(os.path.join(args.output_dir, 'model_best.pt'))
            print(f"  New best model saved! Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            model.save(os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pt'))
    
    # Save final model
    model.save(os.path.join(args.output_dir, 'model_final.pt'))
    
    # Save metrics
    metrics_dict = metrics.to_dict()
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        import json
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nTraining complete. Final metrics:")
    for name, value in metrics_dict.items():
        print(f"  {name}: {value['latest']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Synobotix VLA Model')
    
    # Model arguments
    parser.add_argument('--model_config', type=str, default='config/model.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--quant_config', type=str, default='config/quantization.yaml',
                       help='Path to quantization configuration file')
    
    # Data arguments
    parser.add_argument('--dataset_type', type=str, default='synthetic',
                       choices=['synthetic', 'lerobot'],
                       help='Training data source')
    parser.add_argument('--repo_id', type=str, default='lerobot/pusht',
                       help='LeRobot dataset repo id (with --dataset_type lerobot)')
    
    parser.add_argument('--image_keys', nargs='*', default=[],
                       help='LeRobot camera keys to load as frames (e.g. observation.image)')
    parser.add_argument('--frame_cache_dir', type=str, default=None,
                       help='Directory caching decoded video frames across epochs')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='Sequence length')
    parser.add_argument('--vocab_size', type=int, default=50000,
                       help='Vocabulary size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping')
    
    # Data arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Quantization arguments
    parser.add_argument('--calibrate', action='store_true',
                       help='Calibrate quantizer before training')
    parser.add_argument('--calib_batch_size', type=int, default=100,
                       help='Batch size for calibration')
    parser.add_argument('--num_calib_samples', type=int, default=10000,
                       help='Number of calibration samples')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval in steps')
    parser.add_argument('--save_interval', type=int, default=1,
                       help='Save interval in epochs')
    
    # Hardware
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Set up output directory with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'train_{timestamp}')
    
    train(args)


if __name__ == '__main__':
    main()
