"""
Memory budget test (memo v2, section 2): 32,768 tokens of context in <= 330 Mio,
including a 10% safety margin.

The budget covers the KV context memory (weights are stored separately). The cache
is measured from real tensor bytes on a filled cache, then projected to the target
context length and to all layers (one cache object holds one layer).
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.kv_cache import SlidingWindowKVCache
from src.models.quantizer import AnchorQuantizer

MIB = 1024 * 1024


class TestMemoryBudget(unittest.TestCase):

    def setUp(self):
        config = yaml.safe_load(open(Path(__file__).parent.parent / 'config' / 'model.yaml'))
        model = config['model']
        self.num_layers = model['num_layers']
        self.num_kv_heads = model['num_kv_heads']
        self.head_dim = model['hidden_dim'] // model['num_heads']
        self.max_tokens = model['max_tokens']
        self.vision_window = model['vision_window_size']
        self.budget_mib = config['memory']['target']
        self.overhead = config['memory']['overhead_factor']

        self.quantizer = AnchorQuantizer(bits=model['kv_cache_bits'], dim=self.head_dim,
                                         outlier_threshold=0.05,
                                         calibration_dir=tempfile.mkdtemp())
        self.quantizer.calibrate(torch.randn(500, 20, self.head_dim))

    def _fill(self, num_tokens, token_type):
        cache = SlidingWindowKVCache(max_tokens=self.max_tokens, dim=self.head_dim,
                                     num_heads=self.num_kv_heads,
                                     vision_window_size=self.vision_window,
                                     quantizer=self.quantizer)
        for _ in range(num_tokens):
            k = torch.randn(1, self.num_kv_heads, 1, self.head_dim)
            cache.add(k, torch.randn_like(k), token_type=token_type)
        return cache

    def test_text_cache_is_really_compressed(self):
        cache = self._fill(200, 'text')
        fp16_bytes = 200 * 2 * self.num_kv_heads * self.head_dim * 2
        self.assertLess(cache.memory_bytes(), fp16_bytes / 4)

    def test_32k_context_fits_330_mio(self):
        text_per_token = self._fill(200, 'text').memory_bytes() / 200
        vision = self._fill(self.vision_window, 'vision').memory_bytes()

        total = (text_per_token * (self.max_tokens - self.vision_window) + vision) * self.num_layers
        total_mib = total * self.overhead / MIB
        print(f"\nProjected KV context memory: {total_mib:.1f} Mio (budget {self.budget_mib})")
        self.assertLessEqual(total_mib, self.budget_mib)


if __name__ == '__main__':
    unittest.main()
