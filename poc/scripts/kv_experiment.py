"""
KV-cache quantization experiment (memo v2, axis A1).

Question: how much accuracy does a 2-bit anchor-based KV cache cost compared to an
fp16 cache, and how much memory does it save?

The trained model is evaluated on held-out episodes. For every variant, the K and V
of each layer go through the exact storage round trip of the cache (pack -> unpack)
before attention uses them, so the numbers are those of a model reading a quantized
cache. Quantizers are calibrated offline on training batches, per layer, separately
for K and V.

Usage:
    python scripts/kv_experiment.py --model_path outputs/train_X/model_final.pt \
        --seq_length 1024 --use_cuda
"""

import argparse
import json
import math
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import get_dataloader
from src.models.kv_quant import GroupedKVQuantizer
from src.models.quantizer import AnchorQuantizer
from src.models.vla_model import VLAModel

MIB = 1024 * 1024

# kind 'anchor': our scheme (static offline calibration, outlier channels in fp16)
# kind 'grouped': KIVI / Kitty re-implementation (dynamic per-group statistics)
VARIANTS = [
    {'name': 'fp16 (baseline)', 'kind': 'fp16'},
    {'name': '2-bit anchor, 5% outliers (ours)', 'kind': 'anchor', 'outliers': 0.05},
    {'name': '2-bit anchor, 10% outliers', 'kind': 'anchor', 'outliers': 0.10},
    {'name': '2-bit naive, no outliers', 'kind': 'anchor', 'outliers': 0.0},
    {'name': 'KIVI K2V2', 'kind': 'grouped', 'boost': 0.0, 'sinks': 0},
    {'name': 'KIVI K2V2* (+32 sink tokens)', 'kind': 'grouped', 'boost': 0.0, 'sinks': 32},
    {'name': 'Kitty (12.5% 4-bit key channels)', 'kind': 'grouped', 'boost': 0.125, 'sinks': 32},
    {'name': 'Kitty-Pro (25% 4-bit key channels)', 'kind': 'grouped', 'boost': 0.25, 'sinks': 32},
]


def to_rows(t: torch.Tensor) -> torch.Tensor:
    """[B, kv_heads, T, head_dim] -> [B * T, kv_heads * head_dim]: one channel per (head, dim)."""
    B, H, T, D = t.shape
    return t.transpose(1, 2).reshape(B * T, H * D)


def from_rows(rows: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    B, H, T, D = like.shape
    return rows.reshape(B, T, H, D).transpose(1, 2).to(like.dtype)


def capture_kv(model, loader, device, num_batches, rows_per_batch):
    """K and V rows sampled from training batches, per layer."""
    captured = [[] for _ in model.layers]
    for layer in model.layers:
        layer.attn.kv_capture = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            tokens = batch['text_tokens'].to(device)
            model.delta_net.reset_hidden(tokens.shape[0])
            model(text_tokens=tokens)
            for j, layer in enumerate(model.layers):
                k, v = layer.attn.kv_capture.pop()
                k_rows, v_rows = to_rows(k), to_rows(v)
                pick = torch.randperm(k_rows.shape[0], device=k_rows.device)[:rows_per_batch]
                captured[j].append((k_rows[pick].cpu(), v_rows[pick].cpu()))
    for layer in model.layers:
        layer.attn.kv_capture = None
    return [(torch.cat([k for k, _ in rows]), torch.cat([v for _, v in rows])) for rows in captured]


def calibrate(captured, bits, outlier_fraction, dim, device):
    quantizers = []
    for k_rows, v_rows in captured:
        pair = []
        for rows in (k_rows, v_rows):
            q = AnchorQuantizer(bits=bits, dim=dim, outlier_threshold=outlier_fraction,
                                calibration_dir=tempfile.mkdtemp())
            q.calibrate(rows.unsqueeze(0))
            pair.append(q.to(device))
        quantizers.append(pair)
    return quantizers


def clear_hooks(model):
    for layer in model.layers:
        layer.attn.kv_transform = None
        layer.attn.kv_mix = None


def install(model, quantizers):
    """Make every attention layer read K/V through the cache storage round trip."""
    for i, layer in enumerate(model.layers):
        pair = None if quantizers is None else quantizers[i]

        def transform(k, v, pair=pair):
            if pair is None:  # fp16 cache
                return k.half().to(k.dtype), v.half().to(v.dtype)
            out = []
            for t, q in zip((k, v), pair):
                packed, outliers = q.pack(to_rows(t))
                out.append(from_rows(q.unpack(packed, outliers), t))
            return out[0], out[1]

        layer.attn.kv_transform = transform


def bytes_per_token_per_layer(quantizers, dim):
    """Real bytes one token occupies in one layer's cache (K and V)."""
    if quantizers is None:
        return 2 * dim * 2  # K and V, fp16
    total = 0
    for q in quantizers[0]:
        packed, outliers = q.pack(torch.zeros(1, dim, device=q.scales.device))
        total += packed.numel() * packed.element_size() + outliers.numel() * outliers.element_size()
    return total


def evaluate(model, loader, device, max_batches):
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            tokens = batch['text_tokens'].to(device)
            targets = batch['target_tokens'].to(device)
            model.delta_net.reset_hidden(tokens.shape[0])
            logits = model(text_tokens=tokens)
            valid = targets != -1
            loss_sum += F.cross_entropy(logits[valid], targets[valid], reduction='sum').item()
            correct += (logits[valid].argmax(-1) == targets[valid]).sum().item()
            count += int(valid.sum())
    return math.exp(loss_sum / count), correct / count, count


def main():
    parser = argparse.ArgumentParser(description='KV-cache quantization experiment')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--repo_id', default='lerobot/pusht')
    parser.add_argument('--seq_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cross_episode', action='store_true',
                        help='Long windows spanning several episodes')
    parser.add_argument('--calib_batches', type=int, default=8)
    parser.add_argument('--calib_rows_per_batch', type=int, default=2048)
    parser.add_argument('--eval_batches', type=int, default=None)
    parser.add_argument('--target_tokens', type=int, default=32768)
    parser.add_argument('--group_size', type=int, default=128,
                        help='Tokens per key group for the KIVI / Kitty variants')
    parser.add_argument('--budget_mib', type=float, default=330.0)
    parser.add_argument('--overhead', type=float, default=1.10)
    parser.add_argument('--output_dir', default='kv_experiment')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    model = VLAModel.load(args.model_path).to(device).eval()
    dim = model.num_kv_heads * (model.dim // model.num_heads)

    data = dict(dataset_type='lerobot', repo_id=args.repo_id, seq_length=args.seq_length,
                batch_size=args.batch_size, num_workers=args.num_workers,
                cross_episode=args.cross_episode)
    train_loader = get_dataloader(split='train', **data)
    val_loader = get_dataloader(split='val', **data)

    print('Capturing K/V for calibration...')
    captured = capture_kv(model, train_loader, device, args.calib_batches, args.calib_rows_per_batch)

    kv_heads, head_dim = model.num_kv_heads, model.dim // model.num_heads
    results = []
    for variant in VARIANTS:
        name = variant['name']
        print(f'\n== {name}')
        clear_hooks(model)
        if variant['kind'] == 'grouped':
            quantizer = GroupedKVQuantizer(group_size=args.group_size, boost_ratio=variant['boost'],
                                           sink_tokens=variant['sinks'])
            for layer in model.layers:
                layer.attn.kv_mix = quantizer
            layer_bytes = quantizer.bytes_for_tokens(args.target_tokens, kv_heads, head_dim)
            basis = 'computed from bit widths'
        else:
            quantizers = None if variant['kind'] == 'fp16' else \
                calibrate(captured, 2, variant['outliers'], dim, device)
            install(model, quantizers)
            layer_bytes = bytes_per_token_per_layer(quantizers, dim) * args.target_tokens
            basis = 'measured packed bytes' if quantizers else 'fp16'
        ppl, accuracy, count = evaluate(model, val_loader, device, args.eval_batches)
        projected = layer_bytes * model.num_layers * args.overhead / MIB
        results.append({'variant': name, 'perplexity': ppl, 'accuracy': accuracy,
                        'eval_tokens': count, 'projected_mib': projected, 'memory_basis': basis})
        print(f'perplexity {ppl:.3f} | top-1 accuracy {accuracy:.4f} | '
              f'{projected:.1f} MiB at {args.target_tokens} tokens ({basis})')

    baseline = results[0]
    print('\n' + '=' * 96)
    print(f'{"variant":<38}{"ppl":>9}{"Δppl":>9}{"acc":>9}{"Δacc":>9}{"MiB@32K":>10}{"vs fp16":>9}')
    for r in results:
        print(f'{r["variant"]:<38}{r["perplexity"]:>9.3f}{r["perplexity"] - baseline["perplexity"]:>+9.3f}'
              f'{r["accuracy"]:>9.4f}{r["accuracy"] - baseline["accuracy"]:>+9.4f}'
              f'{r["projected_mib"]:>10.1f}{baseline["projected_mib"] / r["projected_mib"]:>8.1f}x')
    print('Memory: measured from packed bytes for fp16 and our anchor scheme; computed from bit widths '
          '(scales and zero-points included) for KIVI / Kitty.')
    ours = results[1]
    print('=' * 96)
    print(f'Memo criteria: Δppl < 1 point: {"MET" if ours["perplexity"] - baseline["perplexity"] < 1 else "NOT MET"}'
          f' | KV memory <= {args.budget_mib:.0f} MiB at {args.target_tokens} tokens: '
          f'{"MET" if ours["projected_mib"] <= args.budget_mib else "NOT MET"}'
          f' (fp16 needs {baseline["projected_mib"]:.0f} MiB)')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'kv_experiment.json'), 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)


if __name__ == '__main__':
    main()
