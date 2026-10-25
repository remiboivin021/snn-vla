"""
Unit Tests for GQA Implementation
Pivot 1: Architecture GQA native (8 à 16 têtes KV)
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gqa import GQAAttention, GQATransformerBlock


class TestGQAAttention(unittest.TestCase):
    """Test cases for GQAAttention module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 64
        self.dim = 512
        self.num_heads = 8
        self.num_kv_heads = 8
        self.head_dim = self.dim // self.num_heads
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        gqa = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        output = gqa(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_different_kv_heads(self):
        """Test with different number of KV heads."""
        # Test with 4 KV heads (2:1 ratio)
        gqa = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=4
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        output = gqa(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_divisibility_check(self):
        """Test that num_heads must be divisible by num_kv_heads."""
        with self.assertRaises(AssertionError):
            GQAAttention(
                dim=self.dim,
                num_heads=8,
                num_kv_heads=5  # 8 is not divisible by 5
            )
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        gqa = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        memory = gqa.get_memory_usage(self.batch_size, self.seq_len)
        
        # Memory should be positive
        self.assertGreater(memory, 0)
        
        # Memory should increase with sequence length
        memory_long = gqa.get_memory_usage(self.batch_size, self.seq_len * 2)
        self.assertGreater(memory_long, memory)
    
    def test_kv_cache_integration(self):
        """Test integration with KV cache."""
        gqa = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        # Create mock KV cache
        kv_cache = {
            'k': torch.randn(self.batch_size, self.num_heads, 0, self.head_dim),
            'v': torch.randn(self.batch_size, self.num_heads, 0, self.head_dim),
            'update': lambda k, v: (torch.cat([kv_cache['k'], k], dim=2),
                                  torch.cat([kv_cache['v'], v], dim=2))
        }
        
        # This test would need actual KV cache implementation
        # For now, just test without cache
        output = gqa(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        gqa = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)
        output = gqa(x)
        
        # Sum output to get a scalar
        scalar = output.sum()
        scalar.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        
        # Check that GQA parameters have gradients
        for param in gqa.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        gqa = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, self.seq_len, self.dim)
            output = gqa(x)
            self.assertEqual(output.shape, (batch_size, self.seq_len, self.dim))


class TestGQATransformerBlock(unittest.TestCase):
    """Test cases for GQATransformerBlock module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 64
        self.dim = 512
        self.num_heads = 8
        self.num_kv_heads = 8
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        block = GQATransformerBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        output = block(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = GQATransformerBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)
        output = block(x)
        
        scalar = output.sum()
        scalar.backward()
        
        self.assertIsNotNone(x.grad)
        
        for param in block.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_different_dimensions(self):
        """Test with different dimensions."""
        for dim in [256, 512, 1024]:
            block = GQATransformerBlock(
                dim=dim,
                num_heads=8,
                num_kv_heads=8
            )
            
            x = torch.randn(self.batch_size, self.seq_len, dim)
            output = block(x)
            self.assertEqual(output.shape, x.shape)


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency of GQA vs standard attention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 128
        self.dim = 512
        self.num_heads = 8
    
    def test_memory_comparison(self):
        """Compare memory usage between GQA and standard attention."""
        # GQA with same number of heads
        gqa = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads  # Same as Q heads
        )
        
        # Standard multi-head attention
        standard = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Calculate parameter counts
        gqa_params = sum(p.numel() for p in gqa.parameters())
        standard_params = sum(p.numel() for p in standard.parameters())
        
        # GQA should have fewer parameters when num_kv_heads < num_heads
        # But with same number, they should be similar
        print(f"GQA params: {gqa_params}")
        print(f"Standard params: {standard_params}")
        
        # The difference should be within 10%
        ratio = gqa_params / standard_params
        self.assertAlmostEqual(ratio, 1.0, delta=0.1)
    
    def test_gqa_with_fewer_kv_heads(self):
        """Test GQA with fewer KV heads reduces parameters."""
        # GQA with half the KV heads
        gqa_fewer = GQAAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads // 2
        )
        
        # Standard attention
        standard = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        gqa_params = sum(p.numel() for p in gqa_fewer.parameters())
        standard_params = sum(p.numel() for p in standard.parameters())
        
        # GQA should have fewer parameters
        ratio = gqa_params / standard_params
        print(f"GQA params: {gqa_params}")
        print(f"Standard params: {standard_params}")
        print(f"Ratio: {ratio}")
        
        self.assertLess(ratio, 0.9)  # At least 10% reduction


if __name__ == '__main__':
    # Run tests
    print("Running GQA Unit Tests...")
    print("=" * 50)
    unittest.main()
