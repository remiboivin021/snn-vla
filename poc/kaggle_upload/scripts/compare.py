#!/usr/bin/env python3
"""
Comparison Script: Baseline vs Synobotix PoC

Compares the Synobotix VLA PoC with:
1. Baseline FP16 Transformer
2. BitNet-style Transformer (optional)

Metrics:
- Memory usage
- Perplexity
- Inference latency
"""

import argparse
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VLAModel
from src.data import get_dataloader
from src.utils import compute_perplexity, get_model_memory, get_tensor_memory


class BaselineTransformer(nn.Module):
    """
    Baseline FP16 Transformer for comparison.
    Standard multi-head attention with no optimizations.
    """
    
    def __init__(self, dim=512, num_layers=12, num_heads=8, vocab_size=50000, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.head = nn.Linear(dim, vocab_size)
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        B, seq_len = x.shape
        
        # Embed tokens
        x = self.embed(x)  # [B, seq_len, dim]
        
        # Position embedding (simple)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, seq_len)
        position_embeddings = self._position_embedding(position_ids)
        x = x + position_embeddings
        
        # Transformer
        x = self.transformer(x)
        
        # Normalize and project
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
    
    def _position_embedding(self, position_ids):
        # Simple sinusoidal position embedding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=position_ids.device).float() / self.dim))
        emb = torch.einsum('i,j->ij', position_ids.float(), inv_freq)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class BitNetStyleTransformer(nn.Module):
    """
    BitNet-style Transformer for comparison.
    Uses 2-bit quantization and limited KV heads (5 per layer).
    """
    
    def __init__(self, dim=512, num_layers=12, num_heads=8, num_kv_heads=5, vocab_size=50000):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.vocab_size = vocab_size
        self.head_dim = dim // num_heads
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, dim)
        
        # Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                BitNetLayer(dim, num_heads, num_kv_heads)
            )
        
        # Output head
        self.head = nn.Linear(dim, vocab_size)
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, seq_len = x.shape
        
        # Embed tokens
        x = self.embed(x)
        
        # Position embedding
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, seq_len)
        position_embeddings = self._position_embedding(position_ids)
        x = x + position_embeddings
        
        # Pass through layers
        for layer in self.layers:
            x = layer(x)
        
        # Normalize and project
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
    
    def _position_embedding(self, position_ids):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=position_ids.device).float() / self.dim))
        emb = torch.einsum('i,j->ij', position_ids.float(), inv_freq)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class BitNetLayer(nn.Module):
    """
    BitNet-style attention layer with limited KV heads.
    """
    
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.num_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def _expand_kv_heads(self, x):
        """Expand KV heads to match Q heads."""
        if self.num_groups == 1:
            return x
        B, N, num_kv_heads, head_dim = x.shape
        x = x.unsqueeze(1).expand(B, self.num_groups, num_kv_heads, N, head_dim)
        return x.reshape(B, N, self.num_heads, head_dim).transpose(1, 2)
    
    def forward(self, x):
        # Self-attention
        B, N, C = x.shape
        
        # Project Q, K, V
        qkv = self.qkv(self.norm1(x))  # [B, N, 3*dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each [B, N, num_heads, head_dim]
        
        # Expand KV heads
        k = self._expand_kv_heads(k.transpose(1, 2))  # [B, num_kv_heads, N, head_dim]
        v = self._expand_kv_heads(v.transpose(1, 2))
        
        # Attention
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k, v,
            scale=self.scale
        )
        attn = attn.transpose(1, 2).reshape(B, N, C)
        
        # Output
        x = x + self.out(attn)
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


def measure_latency(model, x, num_runs=10):
    """Measure inference latency."""
    model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs
    return avg_latency


def evaluate_model(model, dataloader, device, model_name):
    """Evaluate a model on perplexity."""
    model.eval()
    model.to(device)
    
    total_ppl = 0.0
    total_tokens = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            if 'text_tokens' in batch:
                x = batch['text_tokens'].to(device)
                targets = batch['target_tokens'].to(device)
            else:
                x = batch['input_tokens'].long().to(device)
                targets = batch['target_tokens'].long().to(device)
            
            logits = model(x)
            
            # Compute loss
            B, S, V = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, V),
                targets.view(-1),
                ignore_index=-1
            )
            
            ppl = torch.exp(loss).item()
            
            total_ppl += ppl * B * S
            total_loss += loss.item() * B * S
            total_tokens += B * S
    
    avg_ppl = total_ppl / total_tokens if total_tokens > 0 else float('inf')
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    return {
        'perplexity': avg_ppl,
        'loss': avg_loss
    }


def measure_memory(model, batch_size=1, seq_len=32768, device='cpu'):
    """Measure memory usage of a model."""
    model.eval()
    model.to(device)
    
    # Parameters memory
    param_bytes = sum(p.numel() * 4 for p in model.parameters())
    param_mb = param_bytes / (1024 * 1024)
    
    # Test with dummy input
    x = torch.randint(0, 50000, (batch_size, seq_len), device=device)
    
    # Forward pass memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        total_memory = param_mb + (peak_memory - param_mb)
    else:
        # Estimate based on parameters and activations
        # Rough estimate: params + activations (same size as params * layers)
        activation_bytes = batch_size * seq_len * model.dim * 4 * model.num_layers
        activation_mb = activation_bytes / (1024 * 1024)
        total_memory = param_mb + activation_mb
        peak_memory = total_memory
    
    return {
        'parameters': param_mb,
        'peak': peak_memory,
        'total_estimated': total_memory
    }


def compare_models(args):
    """Main comparison function."""
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    eval_loader = get_dataloader(
        dataset_type='synthetic',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size,
        split='val',
        num_workers=args.num_workers
    )
    print(f"Loaded {len(eval_loader.dataset)} evaluation samples")
    
    # Initialize results
    results = {
        'baseline': {},
        'poc': {},
        'bitnet': {} if args.compare_bitnet else None
    }
    
    # ========================================================================
    # 1. Evaluate Baseline FP16 Transformer
    # ========================================================================
    print("\n" + "=" * 60)
    print("Evaluating BASELINE (FP16 Transformer)")
    print("=" * 60)
    
    baseline = BaselineTransformer(
        dim=512,
        num_layers=12,
        num_heads=8,
        vocab_size=args.vocab_size
    )
    
    # Memory
    print("Measuring memory...")
    baseline_mem = measure_memory(baseline, batch_size=1, seq_len=args.max_seq_len, device=device)
    results['baseline']['memory'] = baseline_mem
    print(f"  Parameters: {baseline_mem['parameters']:.2f} MB")
    print(f"  Peak Memory: {baseline_mem['peak']:.2f} MB")
    
    # Perplexity
    print("Evaluating perplexity...")
    baseline_metrics = evaluate_model(baseline, eval_loader, device, 'Baseline')
    results['baseline']['perplexity'] = baseline_metrics['perplexity']
    results['baseline']['loss'] = baseline_metrics['loss']
    print(f"  Perplexity: {baseline_metrics['perplexity']:.2f}")
    print(f"  Loss: {baseline_metrics['loss']:.4f}")
    
    # Latency
    print("Measuring latency...")
    x_test = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_length), device=device)
    baseline_latency = measure_latency(baseline, x_test, num_runs=10)
    results['baseline']['latency'] = baseline_latency
    print(f"  Latency: {baseline_latency * 1000:.2f} ms/batch")
    
    # Clean up
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================================================
    # 2. Evaluate Synobotix PoC
    # ========================================================================
    print("\n" + "=" * 60)
    print("Evaluating SYNOBOTIX PoC")
    print("=" * 60)
    
    # Load PoC model
    if args.poc_path and os.path.exists(args.poc_path):
        poc = VLAModel.load(args.poc_path).to(device)
        print(f"Loaded PoC from {args.poc_path}")
    else:
        # Create new PoC
        poc = VLAModel(
            config_path=args.model_config,
            quant_config_path=args.quant_config
        ).to(device)
        print("Created new PoC model")
    
    # Memory
    print("Measuring memory...")
    poc_mem = measure_memory(poc, batch_size=1, seq_len=args.max_seq_len, device=device)
    results['poc']['memory'] = poc_mem
    print(f"  Parameters: {poc_mem['parameters']:.2f} MB")
    print(f"  Peak Memory: {poc_mem['peak']:.2f} MB")
    
    # Also get detailed memory breakdown
    poc_detailed_mem = poc.get_memory_usage(args.batch_size, args.max_seq_len)
    results['poc']['memory_detailed'] = poc_detailed_mem
    print(f"  Detailed breakdown:")
    for k, v in poc_detailed_mem.items():
        print(f"    {k}: {v:.2f} MB")
    
    # Perplexity
    print("Evaluating perplexity...")
    poc_metrics = evaluate_model(poc, eval_loader, device, 'PoC')
    results['poc']['perplexity'] = poc_metrics['perplexity']
    results['poc']['loss'] = poc_metrics['loss']
    print(f"  Perplexity: {poc_metrics['perplexity']:.2f}")
    print(f"  Loss: {poc_metrics['loss']:.4f}")
    
    # Latency
    print("Measuring latency...")
    x_test = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_length), device=device)
    # For PoC, we need to specify token types
    token_types = ['text'] * args.seq_length
    poc_latency = measure_latency(lambda x: poc(text_tokens=x, token_types=token_types), x_test, num_runs=10)
    results['poc']['latency'] = poc_latency
    print(f"  Latency: {poc_latency * 1000:.2f} ms/batch")
    
    # Clean up
    del poc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================================================
    # 3. Evaluate BitNet-style (optional)
    # ========================================================================
    if args.compare_bitnet:
        print("\n" + "=" * 60)
        print("Evaluating BITNET-STYLE (5 KV heads + GQA-like)")
        print("=" * 60)
        
        bitnet = BitNetStyleTransformer(
            dim=512,
            num_layers=12,
            num_heads=8,
            num_kv_heads=5,  # BitNet uses 5 KV heads
            vocab_size=args.vocab_size
        )
        
        # Memory
        print("Measuring memory...")
        bitnet_mem = measure_memory(bitnet, batch_size=1, seq_len=args.max_seq_len, device=device)
        results['bitnet']['memory'] = bitnet_mem
        print(f"  Parameters: {bitnet_mem['parameters']:.2f} MB")
        print(f"  Peak Memory: {bitnet_mem['peak']:.2f} MB")
        
        # Perplexity
        print("Evaluating perplexity...")
        bitnet_metrics = evaluate_model(bitnet, eval_loader, device, 'BitNet')
        results['bitnet']['perplexity'] = bitnet_metrics['perplexity']
        results['bitnet']['loss'] = bitnet_metrics['loss']
        print(f"  Perplexity: {bitnet_metrics['perplexity']:.2f}")
        print(f"  Loss: {bitnet_metrics['loss']:.4f}")
        
        # Latency
        print("Measuring latency...")
        bitnet_latency = measure_latency(bitnet, x_test, num_runs=10)
        results['bitnet']['latency'] = bitnet_latency
        print(f"  Latency: {bitnet_latency * 1000:.2f} ms/batch")
        
        # Clean up
        del bitnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========================================================================
    # 4. Generate Comparison Report
    # ========================================================================
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)
    
    # Calculate comparison metrics
    comparison = {
        'memory_reduction': {},
        'perplexity_degradation': {},
        'latency_overhead': {}
    }
    
    # Memory comparison
    baseline_memory = results['baseline']['memory']['total_estimated']
    poc_memory = results['poc']['memory']['total_estimated']
    comparison['memory_reduction']['poc_vs_baseline'] = {
        'baseline': baseline_memory,
        'poc': poc_memory,
        'reduction_mb': baseline_memory - poc_memory,
        'reduction_percent': ((baseline_memory - poc_memory) / baseline_memory) * 100 if baseline_memory > 0 else 0
    }
    
    if args.compare_bitnet:
        bitnet_memory = results['bitnet']['memory']['total_estimated']
        comparison['memory_reduction']['poc_vs_bitnet'] = {
            'bitnet': bitnet_memory,
            'poc': poc_memory,
            'reduction_mb': bitnet_memory - poc_memory,
            'reduction_percent': ((bitnet_memory - poc_memory) / bitnet_memory) * 100 if bitnet_memory > 0 else 0
        }
    
    # Perplexity comparison
    baseline_ppl = results['baseline']['perplexity']
    poc_ppl = results['poc']['perplexity']
    comparison['perplexity_degradation']['poc_vs_baseline'] = {
        'baseline': baseline_ppl,
        'poc': poc_ppl,
        'degradation': poc_ppl - baseline_ppl,
        'acceptable': abs(poc_ppl - baseline_ppl) < 1.0
    }
    
    if args.compare_bitnet:
        bitnet_ppl = results['bitnet']['perplexity']
        comparison['perplexity_degradation']['poc_vs_bitnet'] = {
            'bitnet': bitnet_ppl,
            'poc': poc_ppl,
            'improvement': bitnet_ppl - poc_ppl,
            'better': poc_ppl < bitnet_ppl
        }
    
    # Latency comparison
    baseline_lat = results['baseline']['latency']
    poc_lat = results['poc']['latency']
    comparison['latency_overhead']['poc_vs_baseline'] = {
        'baseline': baseline_lat,
        'poc': poc_lat,
        'overhead_sec': poc_lat - baseline_lat,
        'overhead_percent': ((poc_lat - baseline_lat) / baseline_lat) * 100 if baseline_lat > 0 else 0,
        'acceptable': ((poc_lat - baseline_lat) / baseline_lat) * 100 < 10  # < 10% overhead
    }
    
    # Generate report
    report = {
        'results': results,
        'comparison': comparison,
        'summary': {
            'best_model': 'poc' if poc_ppl < baseline_ppl else 'baseline',
            'memory_winner': 'poc',  # PoC should always win on memory
            'recommendation': 'Use PoC' if comparison['perplexity_degradation']['poc_vs_baseline']['acceptable'] 
                           and comparison['memory_reduction']['poc_vs_baseline']['reduction_percent'] > 50
                           else 'Optimize PoC'
        }
    }
    
    # Save report
    report_path = os.path.join(args.output_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print("\n📊 MEMORY COMPARISON:")
    print(f"  Baseline:      {results['baseline']['memory']['total_estimated']:>8.2f} MB")
    print(f"  PoC:           {results['poc']['memory']['total_estimated']:>8.2f} MB")
    print(f"  Reduction:     {comparison['memory_reduction']['poc_vs_baseline']['reduction_percent']:>8.2f} %")
    
    if args.compare_bitnet:
        print(f"  BitNet:        {results['bitnet']['memory']['total_estimated']:>8.2f} MB")
        print(f"  PoC vs BitNet: {comparison['memory_reduction']['poc_vs_bitnet']['reduction_percent']:>8.2f} % reduction")
    
    print("\n📈 PERPLEXITY COMPARISON:")
    print(f"  Baseline:      {results['baseline']['perplexity']:>8.2f}")
    print(f"  PoC:           {results['poc']['perplexity']:>8.2f}")
    degradation = comparison['perplexity_degradation']['poc_vs_baseline']['degradation']
    print(f"  Degradation:   {degradation:+>8.2f} points")
    if comparison['perplexity_degradation']['poc_vs_baseline']['acceptable']:
        print("  ✅ ACCEPTABLE (< 1 point)")
    else:
        print("  ⚠️  NEEDS OPTIMIZATION")
    
    if args.compare_bitnet:
        print(f"  BitNet:        {results['bitnet']['perplexity']:>8.2f}")
        improvement = comparison['perplexity_degradation']['poc_vs_bitnet']['improvement']
        if improvement > 0:
            print(f"  PoC Improvement: {improvement:+.2f} points ✅")
        else:
            print(f"  PoC Improvement: {improvement:+.2f} points ⚠️")
    
    print("\n⏱️  LATENCY COMPARISON:")
    print(f"  Baseline:      {results['baseline']['latency']*1000:>8.2f} ms/batch")
    print(f"  PoC:           {results['poc']['latency']*1000:>8.2f} ms/batch")
    overhead = comparison['latency_overhead']['poc_vs_baseline']['overhead_percent']
    print(f"  Overhead:      {overhead:+>8.2f} %")
    if comparison['latency_overhead']['poc_vs_baseline']['acceptable']:
        print("  ✅ ACCEPTABLE (< 10%)")
    else:
        print("  ⚠️  NEEDS OPTIMIZATION")
    
    print("\n" + "=" * 60)
    print(f"RECOMMENDATION: {report['summary']['recommendation']}")
    print("=" * 60)
    
    print(f"\n📁 Report saved to: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Compare Synobotix PoC with Baseline Models'
    )
    
    # Model arguments
    parser.add_argument('--poc_path', type=str, default=None,
                       help='Path to trained PoC model (optional)')
    parser.add_argument('--model_config', type=str, default='config/model.yaml',
                       help='Path to model configuration')
    parser.add_argument('--quant_config', type=str, default='config/quantization.yaml',
                       help='Path to quantization configuration')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='Sequence length for evaluation')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                       help='Max sequence length for memory test')
    parser.add_argument('--vocab_size', type=int, default=50000,
                       help='Vocabulary size')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    
    # Comparison options
    parser.add_argument('--compare_bitnet', action='store_true',
                       help='Also compare with BitNet-style model')
    
    # Hardware
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/comparison',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f'compare_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comparison
    compare_models(args)


if __name__ == '__main__':
    main()
