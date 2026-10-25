"""VLAModel.get_memory_usage() must report measured bytes, not a formula."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vla_model import VLAModel

MIB = 1024 * 1024


class TestModelMemory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = VLAModel()

    def test_parameters_are_measured(self):
        expected = sum(p.numel() * p.element_size() for p in self.model.parameters()) / MIB
        self.assertAlmostEqual(self.model.get_memory_usage()['parameters'], expected)

    def test_kv_cache_follows_content(self):
        cache = self.model.kv_cache
        cache.clear()
        self.assertEqual(self.model.get_memory_usage()['kv_cache'], 0.0)

        k = torch.randn(1, cache.num_heads, 50, cache.dim)
        cache.add(k, torch.randn_like(k), token_type='text')
        measured = self.model.get_memory_usage()['kv_cache']
        # 50 tokens, K and V, fp16
        self.assertAlmostEqual(measured, 50 * 2 * cache.num_heads * cache.dim * 2 / MIB)
        cache.clear()

    def test_total_is_sum_of_measured_parts(self):
        usage = self.model.get_memory_usage()
        parts = usage['parameters'] + usage['buffers'] + usage['kv_cache']
        self.assertAlmostEqual(usage['total'], parts)
        self.assertGreater(usage['process_rss'], 0)


if __name__ == '__main__':
    unittest.main()
