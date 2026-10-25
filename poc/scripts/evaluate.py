"""
Evaluation Script for Synobotix VLA PoC

Evaluates model performance on various metrics and tasks.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VLAModel
from src.data import get_dataloader, get_robotic_dataloader
from src.utils import MetricsTracker, compute_perplexity, compute_accuracy, RoboticTaskMetrics


def evaluate_perplexity(args):
    """
    Evaluate model perplexity on a dataset.
    """
    print("Evaluating Perplexity...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = VLAModel.load(args.model_path).to(device)
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    
    # Load data
    print("\nLoading evaluation data...")
    if args.dataset_type == 'synthetic':
        dataloader = get_dataloader(
            dataset_type='synthetic',
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            split='test'
        )
    else:
        dataloader = get_robotic_dataloader(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            split='test'
        )
    
    # Evaluation
    total_loss = 0.0
    total_ppl = 0.0
    total_tokens = 0
    
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.dataset_type == 'synthetic':
                text_tokens = batch['text_tokens'].to(device)
                targets = batch['target_tokens'].to(device)
            else:
                # For robotic dataset
                text_tokens = batch['input_tokens'].long().to(device)
                targets = batch['target_tokens'].long().to(device)
            
            # Forward pass
            logits = model(text_tokens=text_tokens)
            
            # Compute loss and perplexity
            batch_size = text_tokens.shape[0]
            seq_len = text_tokens.shape[1]
            vocab_size = logits.shape[-1]
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            
            ppl = compute_perplexity(logits, targets)
            
            total_loss += loss.item() * batch_size * seq_len
            total_ppl += ppl * batch_size * seq_len
            total_tokens += batch_size * seq_len
            
            metrics.add('loss', loss.item())
            metrics.add('perplexity', ppl)
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                      f"Loss={loss.item():.4f}, PPL={ppl:.2f}")
    
    avg_loss = total_loss / total_tokens
    avg_ppl = total_ppl / total_tokens
    
    print(f"\nEvaluation Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average Perplexity: {avg_ppl:.2f}")
    print(f"  Total Tokens: {total_tokens:,}")
    
    # Save results
    results = {
        'average_loss': avg_loss,
        'average_perplexity': avg_ppl,
        'total_tokens': total_tokens,
        'metrics': metrics.to_dict()
    }
    
    output_path = os.path.join(args.output_dir, 'eval_perplexity.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


def evaluate_memory(args):
    """
    Evaluate model memory usage at different sequence lengths.
    """
    print("Evaluating Memory Usage...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = VLAModel.load(args.model_path).to(device)
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    
    # Test different sequence lengths
    seq_lengths = args.seq_lengths or [64, 128, 256, 512, 1024]
    
    results = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting with sequence length: {seq_len}")
        
        # Create dummy input
        text_tokens = torch.randint(0, 50000, (args.batch_size, seq_len))
        
        # Warm up
        with torch.no_grad():
            _ = model(text_tokens=text_tokens.to(device))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory
        with torch.no_grad():
            # Forward pass
            logits = model(text_tokens=text_tokens.to(device))
            
            # Get memory usage
            memory_usage = model.get_memory_usage(args.batch_size, seq_len)
            
            # Also get peak memory
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                peak_memory = 0.0
        
        memory_usage['peak'] = peak_memory
        
        results.append({
            'sequence_length': seq_len,
            'memory_usage': memory_usage,
            'logits_shape': logits.shape
        })
        
        print(f"  Parameters: {memory_usage.get('parameters', 0):.2f} MB")
        print(f"  KV Cache: {memory_usage.get('kv_cache', 0):.2f} MB")
        print(f"  Activations: {memory_usage.get('activations', 0):.2f} MB")
        print(f"  DeltaNet: {memory_usage.get('delta_net', 0):.2f} MB")
        print(f"  Total: {memory_usage.get('total', 0):.2f} MB")
        print(f"  Peak: {peak_memory:.2f} MB")
    
    # Save results
    output_path = os.path.join(args.output_dir, 'eval_memory.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMemory results saved to {output_path}")
    
    return results


def evaluate_robotic_tasks(args):
    """
    Evaluate model on robotic manipulation tasks.
    """
    print("Evaluating Robotic Tasks...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = VLAModel.load(args.model_path).to(device)
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    
    # Load robotic data
    print("\nLoading robotic evaluation data...")
    dataloader = get_robotic_dataloader(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        split='test'
    )
    
    # Task metrics
    task_metrics = RoboticTaskMetrics()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_tokens = batch['input_tokens'].long().to(device)
            target_tokens = batch['target_tokens'].long().to(device)
            tasks = batch['tasks']
            token_types = batch['token_types']
            
            # Forward pass
            logits = model(text_tokens=input_tokens)
            
            # Get predictions
            batch_size = input_tokens.shape[0]
            seq_len = input_tokens.shape[1]
            
            preds = torch.argmax(logits, dim=-1)  # [B, S]
            
            # Calculate accuracy by token type
            for token_type in ['text', 'vision']:
                # Create mask for this token type
                type_mask = torch.tensor([1 if t == token_type else 0 
                                         for t in token_types], device=device)
                type_mask = type_mask.view(batch_size, seq_len)
                
                if type_mask.sum() > 0:
                    masked_preds = preds[type_mask == 1]
                    masked_targets = target_tokens[type_mask == 1]
                    
                    acc = (masked_preds == masked_targets).float().mean().item()
                    task_metrics.add_result(
                        task_name=token_type,
                        success=acc > 0.5,  # Arbitrary threshold
                        completion_time=0.0,  # Not measured in PoC
                        precision=acc
                    )
            
            # Overall accuracy
            overall_acc = (preds == target_tokens).float().mean().item()
            task_metrics.add_result(
                task_name='overall',
                success=overall_acc > 0.5,
                completion_time=0.0,
                precision=overall_acc
            )
            
            if (batch_idx + 1) % args.log_interval == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                      f"Overall Acc={overall_acc:.4f}")
    
    # Print results
    print(f"\nRobotic Task Evaluation Results:")
    print(f"  Text Token Accuracy: {task_metrics.get_avg_precision('text'):.4f}")
    print(f"  Vision Token Accuracy: {task_metrics.get_avg_precision('vision'):.4f}")
    print(f"  Overall Accuracy: {task_metrics.get_avg_precision('overall'):.4f}")
    print(f"  Text Success Rate: {task_metrics.get_success_rate('text'):.4f}")
    print(f"  Vision Success Rate: {task_metrics.get_success_rate('vision'):.4f}")
    
    # Save results
    results = {
        'text_accuracy': task_metrics.get_avg_precision('text'),
        'vision_accuracy': task_metrics.get_avg_precision('vision'),
        'overall_accuracy': task_metrics.get_avg_precision('overall'),
        'text_success_rate': task_metrics.get_success_rate('text'),
        'vision_success_rate': task_metrics.get_success_rate('vision'),
        'total_tasks': len(task_metrics.task_results)
    }
    
    output_path = os.path.join(args.output_dir, 'eval_robotic.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


def evaluate_all(args):
    """
    Run all evaluation types.
    """
    print("Running Full Evaluation...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = Path(args.output_dir).name if args.output_dir else "eval"
    
    results = {}
    
    # Perplexity evaluation
    print("\n" + "=" * 50)
    print("1. Perplexity Evaluation")
    print("=" * 50)
    results['perplexity'] = evaluate_perplexity(args)
    
    # Memory evaluation
    print("\n" + "=" * 50)
    print("2. Memory Evaluation")
    print("=" * 50)
    results['memory'] = evaluate_memory(args)
    
    # Robotic tasks evaluation
    print("\n" + "=" * 50)
    print("3. Robotic Tasks Evaluation")
    print("=" * 50)
    results['robotic'] = evaluate_robotic_tasks(args)
    
    # Save all results
    all_results_path = os.path.join(args.output_dir, 'eval_all.json')
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll results saved to {all_results_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Perplexity: {results['perplexity']['average_perplexity']:.2f}")
    print(f"Memory at max seq_len: {results['memory'][-1]['memory_usage']['total']:.2f} MB")
    print(f"Robotic Overall Accuracy: {results['robotic']['overall_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Synobotix VLA Model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    
    # Evaluation type
    parser.add_argument('--eval_type', type=str, choices=['perplexity', 'memory', 'robotic', 'all'],
                       default='all', help='Type of evaluation to run')
    
    # Data arguments
    parser.add_argument('--dataset_type', type=str, default='synthetic',
                       choices=['synthetic', 'robotic'], help='Dataset type for evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='Sequence length')
    parser.add_argument('--vocab_size', type=int, default=50000,
                       help='Vocabulary size')
    parser.add_argument('--seq_lengths', nargs='+', type=int, default=None,
                       help='Sequence lengths to test for memory evaluation')
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval')
    
    # Hardware
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    if args.eval_type != 'all':
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(args.output_dir, f'{args.eval_type}_{timestamp}')
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(args.output_dir, f'full_eval_{timestamp}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run appropriate evaluation
    if args.eval_type == 'perplexity':
        evaluate_perplexity(args)
    elif args.eval_type == 'memory':
        evaluate_memory(args)
    elif args.eval_type == 'robotic':
        evaluate_robotic_tasks(args)
    else:
        evaluate_all(args)


if __name__ == '__main__':
    import datetime
    main()
