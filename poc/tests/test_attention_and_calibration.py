"""
Regression tests for the two defects that made an accuracy experiment meaningless:
- attention had no causal mask, so a position could read the token it must predict;
- the 2-bit calibration reconstructed every positive value as <= 0.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gqa import GQAAttention
from src.models.quantizer import AnchorQuantizer


class TestCausalAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.attn = GQAAttention(dim=64, num_heads=4, num_kv_heads=2).eval()

    def test_future_tokens_do_not_change_past_outputs(self):
        x = torch.randn(1, 10, 64)
        changed = x.clone()
        changed[:, 6:] = torch.randn(1, 4, 64)
        out, out_changed = self.attn(x), self.attn(changed)
        self.assertTrue(torch.allclose(out[:, :6], out_changed[:, :6], atol=1e-5))
        self.assertFalse(torch.allclose(out[:, 6:], out_changed[:, 6:], atol=1e-5))

    def test_position_matters(self):
        x = torch.randn(1, 8, 64)
        swapped = x.clone()
        swapped[:, [2, 3]] = x[:, [3, 2]]
        # Same set of tokens before position 5, different order: outputs must differ
        self.assertFalse(torch.allclose(self.attn(x)[:, 5], self.attn(swapped)[:, 5], atol=1e-5))

    def test_kv_transform_is_applied_to_cached_kv(self):
        x = torch.randn(1, 6, 64)
        reference = self.attn(x)
        self.attn.kv_transform = lambda k, v: (torch.zeros_like(k), torch.zeros_like(v))
        self.assertFalse(torch.allclose(self.attn(x), reference, atol=1e-5))
        self.attn.kv_transform = lambda k, v: (k, v)
        self.assertTrue(torch.allclose(self.attn(x), reference, atol=1e-6))


class TestTwoBitCalibration(unittest.TestCase):

    def test_positive_values_survive(self):
        torch.manual_seed(0)
        quantizer = AnchorQuantizer(bits=2, dim=64, outlier_threshold=0.05,
                                    calibration_dir=tempfile.mkdtemp())
        quantizer.calibrate(torch.randn(500, 20, 64))
        x = torch.randn(8, 20, 64)
        reconstructed, _ = quantizer.quantize(x)
        regular = ~quantizer.outlier_mask
        relative_mse = (((reconstructed - x) ** 2)[..., regular].mean() / (x[..., regular] ** 2).mean())
        # 4-level uniform quantizer on a Gaussian: ~0.12 at best; the defect gave > 0.5
        self.assertLess(relative_mse.item(), 0.2)
        self.assertGreater(reconstructed[..., regular][x[..., regular] > 0.5].min().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
