"""Padding targets (-1) must not reach the loss kernels: on CUDA they crash with a device-side assert."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import compute_accuracy, compute_perplexity


class TestPaddingIgnored(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(2, 6, 10)
        self.targets = torch.randint(0, 10, (2, 6))
        self.padded = self.targets.clone()
        self.padded[:, 4:] = -1

    def test_perplexity_ignores_padding(self):
        expected = compute_perplexity(self.logits[:, :4].contiguous(), self.targets[:, :4].contiguous())
        self.assertAlmostEqual(compute_perplexity(self.logits, self.padded), expected, places=4)

    def test_accuracy_ignores_padding(self):
        for top_k in (1, 3):
            expected = compute_accuracy(self.logits[:, :4].contiguous(), self.targets[:, :4].contiguous(), top_k)
            self.assertAlmostEqual(compute_accuracy(self.logits, self.padded, top_k), expected, places=6)


if __name__ == '__main__':
    unittest.main()
