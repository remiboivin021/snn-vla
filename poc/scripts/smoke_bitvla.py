"""
BitVLA smoke test: does quantizing the K/V of BitVLA's attention change the actions it predicts?

BitVLA (github.com/ustcwhy/BitVLA) has no real KV cache at inference: one bidirectional
forward over ~600 tokens produces the 8-action chunk. This test therefore measures the
accuracy impact of storing K/V in low precision (K and V pass through the quantizer after
RoPE, before attention reads them), NOT the 32K-token memory budget.

No simulator: the model predicts one action chunk on a recorded LIBERO observation, once per
variant. The first two variants (fp16, fp16 repeat) give the noise floor.

Usage (BitVLA cloned at --bitvla_dir, Python 3.10 environment of the fork):
    python scripts/smoke_bitvla.py --bitvla_dir bitvla_src --checkpoint ckpt_dir \
        --observation bitvla_src/openvla-oft/experiments/robot/libero/sample_libero_spatial_observation.pkl
"""

import argparse
import copy
import importlib.util
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_ANCHOR = '        key_states = repeat_kv(key_states, self.num_key_value_groups)\n'
PATCH_LINE = ('        if KV_HOOK is not None:\n'
              '            key_states, value_states = KV_HOOK(key_states, value_states)\n')


def load_kv_quant():
    """Load GroupedKVQuantizer without importing the poc `src.models` package (needs torch only)."""
    path = os.path.join(os.path.dirname(HERE), 'src', 'models', 'kv_quant.py')
    spec = importlib.util.spec_from_file_location('kv_quant', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GroupedKVQuantizer


def patch_bitnet_attention(bitvla_dir: str) -> str:
    """Add a module-level KV_HOOK to BitNetAttention.forward (K/V after RoPE, before repeat_kv)."""
    path = os.path.join(bitvla_dir, 'transformers', 'src', 'transformers', 'models', 'llava', 'modeling_bitnet.py')
    with open(path) as f:
        source = f.read()
    if 'KV_HOOK' in source:
        return path
    assert source.count(PATCH_ANCHOR) == 1, 'BitNetAttention anchor not found exactly once'
    source = source.replace(PATCH_ANCHOR, PATCH_LINE + PATCH_ANCHOR)
    assert source.count('\nclass BitNetAttention(') == 1, 'BitNetAttention class not found exactly once'
    source = source.replace('\nclass BitNetAttention(', '\nKV_HOOK = None\n\n\nclass BitNetAttention(')
    with open(path, 'w') as f:
        f.write(source)
    return path


def make_hook(quantizer, sink_tokens):
    """K/V [B, kv_heads, T, head_dim] -> what attention would read back from a quantized store."""
    def hook(k, v):
        k_q, v_q = quantizer.quantize_keys(k), quantizer.quantize_values(v)
        if sink_tokens:
            k_q[:, :, :sink_tokens], v_q[:, :, :sink_tokens] = k[:, :, :sink_tokens], v[:, :, :sink_tokens]
        return k_q.to(k.dtype), v_q.to(v.dtype)
    return hook


def main():
    parser = argparse.ArgumentParser(description='BitVLA KV quantization smoke test')
    parser.add_argument('--bitvla_dir', required=True, help='Clone of ustcwhy/BitVLA')
    parser.add_argument('--checkpoint', required=True, help='Local dir of a BitVLA LIBERO checkpoint')
    parser.add_argument('--observation', required=True, help='sample_libero_*_observation.pkl')
    parser.add_argument('--task_suite', default='libero_spatial')
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--output', default='smoke_bitvla.json')
    args = parser.parse_args()

    modeling_path = patch_bitnet_attention(args.bitvla_dir)
    print('patched', modeling_path)
    oft = os.path.join(args.bitvla_dir, 'openvla-oft')
    sys.path[:0] = [HERE, oft, os.path.join(oft, 'bitvla')]
    import stub_heavy  # noqa: F401  (tensorflow stack, training data only)

    from transformers.models.llava import modeling_bitnet
    from experiments.robot.bitnet_utils import get_bitnet_vla, get_bitnet_vla_action
    from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector

    cfg = SimpleNamespace(pretrained_checkpoint=args.checkpoint, load_in_8bit=False, load_in_4bit=False,
                          use_l1_regression=True, use_diffusion=False, use_film=False,
                          num_images_in_input=2, use_proprio=True, center_crop=True,
                          num_open_loop_steps=8, unnorm_key=args.task_suite, model_family='bitnet')
    vla = get_bitnet_vla(cfg)
    if cfg.unnorm_key not in vla.norm_stats and f'{cfg.unnorm_key}_no_noops' in vla.norm_stats:
        cfg.unnorm_key += '_no_noops'
    hidden = vla.config.text_config.hidden_size
    proprio_projector = get_proprio_projector(cfg, hidden, proprio_dim=8)
    action_head = get_action_head(cfg, hidden)
    processor = get_processor(cfg)

    with open(args.observation, 'rb') as f:
        sample = pickle.load(f)
    task = sample['task_description']

    def predict():
        obs = copy.deepcopy({k: sample[k] for k in ('full_image', 'wrist_image', 'state')})
        return np.stack(get_bitnet_vla_action(cfg, vla, processor, obs, task, action_head, proprio_projector))

    Quantizer = load_kv_quant()
    variants = [
        ('fp16 (reference)', None),
        ('fp16 repeat (noise floor)', None),
        ('4-bit K/V (group-wise)', (Quantizer(args.group_size, boost_ratio=0.0, sink_tokens=0, bits=4), 0)),
        ('KIVI K2V2', (Quantizer(args.group_size, boost_ratio=0.0, sink_tokens=0), 0)),
        ('Kitty (12.5% 4-bit key channels, 32 sinks)', (Quantizer(args.group_size, 0.125, 32), 32)),
        ('Kitty-Pro (25% 4-bit key channels, 32 sinks)', (Quantizer(args.group_size, 0.25, 32), 32)),
    ]

    actions = {}
    for name, spec in variants:
        modeling_bitnet.KV_HOOK = None if spec is None else make_hook(*spec)
        actions[name] = predict()
        print(f'{name}: first action {np.round(actions[name][0], 3).tolist()}', flush=True)
    modeling_bitnet.KV_HOOK = None

    reference = actions['fp16 (reference)']
    scale = np.abs(reference).mean()
    rows = []
    print(f'\ntask: {task}\n{"variant":<48}{"mean |Δaction|":>16}{"max |Δaction|":>16}{"relative":>10}')
    for name, _ in variants[1:]:
        diff = np.abs(actions[name] - reference)
        rows.append({'variant': name, 'mean_abs_diff': float(diff.mean()), 'max_abs_diff': float(diff.max()),
                     'relative_to_mean_abs_action': float(diff.mean() / scale)})
        print(f'{name:<48}{diff.mean():>16.5f}{diff.max():>16.5f}{diff.mean() / scale:>10.3f}')

    with open(args.output, 'w') as f:
        json.dump({'task': task, 'checkpoint': args.checkpoint, 'reference_action_chunk': reference.tolist(),
                   'results': rows}, f, indent=2)


if __name__ == '__main__':
    main()
