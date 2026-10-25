"""KIVI / Kitty style quantizer: behaviour the comparison relies on."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gqa import GQAAttention
from src.models.kv_quant import GroupedKVQuantizer


def keys_with_heavy_channels(B=1, H=2, T=256, D=64):
    torch.manual_seed(0)
    k = torch.randn(B, H, T, D)
    k[..., :4] *= 8.0  # a few channels dominate, as in real keys
    return k


class TestGroupedKVQuantizer(unittest.TestCase):

    def test_boosting_channels_lowers_key_error(self):
        k = keys_with_heavy_channels()
        kivi = GroupedKVQuantizer(group_size=64, boost_ratio=0.0).quantize_keys(k)
        kitty = GroupedKVQuantizer(group_size=64, boost_ratio=0.125).quantize_keys(k)
        self.assertLess(((kitty - k) ** 2).mean(), ((kivi - k) ** 2).mean())

    def test_key_groups_are_independent(self):
        quantizer = GroupedKVQuantizer(group_size=64, boost_ratio=0.125)
        k = keys_with_heavy_channels()
        changed = k.clone()
        changed[:, :, 128:192] = torch.randn_like(changed[:, :, 128:192]) * 3
        self.assertTrue(torch.equal(quantizer.quantize_keys(k)[:, :, :128],
                                    quantizer.quantize_keys(changed)[:, :, :128]))

    def test_values_are_quantized_per_token(self):
        quantizer = GroupedKVQuantizer(group_size=64)
        v = torch.randn(1, 2, 128, 64)
        changed = v.clone()
        changed[:, :, 10] = torch.randn(1, 2, 64) * 5
        same = torch.ones(128, dtype=torch.bool)
        same[10] = False
        self.assertTrue(torch.equal(quantizer.quantize_values(v)[:, :, same],
                                    quantizer.quantize_values(changed)[:, :, same]))

    def test_partial_last_group_stays_exact(self):
        quantizer = GroupedKVQuantizer(group_size=64)
        k = torch.randn(1, 1, 100, 16)
        out = quantizer.quantize_keys(k)
        self.assertTrue(torch.equal(out[:, :, 64:], k[:, :, 64:]))
        self.assertFalse(torch.equal(out[:, :, :64], k[:, :, :64]))

    def test_exact_mask_covers_sinks_and_current_group_only(self):
        mask = GroupedKVQuantizer(group_size=4, sink_tokens=2).exact_mask(12, 'cpu')
        self.assertTrue(mask[9, :2].all())            # sinks, seen by every query
        self.assertTrue(mask[9, 8:].all())            # group being filled
        self.assertFalse(mask[9, 2:8].any())          # older groups are quantized

    def test_memory_ordering(self):
        kivi = GroupedKVQuantizer(boost_ratio=0.0).bytes_for_tokens(32768, 8, 64)
        kitty = GroupedKVQuantizer(boost_ratio=0.125).bytes_for_tokens(32768, 8, 64)
        fp16 = 32768 * 2 * 8 * 64 * 2
        self.assertLess(kivi, kitty)
        self.assertLess(kitty, fp16 / 4)


class TestMixedAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.attn = GQAAttention(dim=64, num_heads=4, num_kv_heads=2).eval()
        self.x = torch.randn(1, 12, 64)
        self.reference = self.attn(self.x)

    def test_all_exact_equals_plain_attention(self):
        self.attn.kv_mix = lambda k, v: (torch.zeros_like(k), torch.zeros_like(v),
                                         torch.ones(k.shape[2], k.shape[2], dtype=torch.bool))
        self.assertTrue(torch.allclose(self.attn(self.x), self.reference, atol=1e-5))

    def test_none_exact_equals_transform_path(self):
        quantizer = GroupedKVQuantizer(group_size=4, boost_ratio=0.25, sink_tokens=0)

        def mix(k, v):
            k_q, v_q, _ = quantizer(k, v)
            return k_q, v_q, torch.zeros(k.shape[2], k.shape[2], dtype=torch.bool)

        self.attn.kv_mix = mix
        mixed = self.attn(self.x)
        self.attn.kv_mix = None
        self.attn.kv_transform = lambda k, v: quantizer(k, v)[:2]
        self.assertTrue(torch.allclose(mixed, self.attn(self.x), atol=1e-5))

    def test_mixed_attention_stays_causal(self):
        quantizer = GroupedKVQuantizer(group_size=4, sink_tokens=2)
        self.attn.kv_mix = quantizer
        changed = self.x.clone()
        changed[:, 8:] = torch.randn(1, 4, 64)
        self.assertTrue(torch.allclose(self.attn(self.x)[:, :8], self.attn(changed)[:, :8], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
