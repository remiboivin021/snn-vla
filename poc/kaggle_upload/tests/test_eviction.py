"""
Unit Tests for Asymmetric Eviction Policy
Pivot 3: Éviction Asymétrique Spatio-Temporelle
"""

import unittest
import torch
import sys
from pathlib import Path
from collections import deque

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.kv_cache import KVCache, SlidingWindowKVCache, AsymmetricEvictionPolicy


class TestAsymmetricEvictionPolicy(unittest.TestCase):
    """Test cases for AsymmetricEvictionPolicy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vision_window_size = 20
        self.max_tokens = 100
        self.policy = AsymmetricEvictionPolicy(
            vision_window_size=self.vision_window_size,
            max_tokens=self.max_tokens
        )
    
    def test_initialization(self):
        """Test policy initialization."""
        self.assertEqual(self.policy.vision_window_size, self.vision_window_size)
        self.assertEqual(self.policy.max_tokens, self.max_tokens)
        self.assertEqual(len(self.policy.vision_window), 0)
        self.assertEqual(len(self.policy.token_types), 0)
    
    def test_add_to_window(self):
        """Test adding tokens to vision window."""
        # Add a vision token
        self.policy.token_types = ['text', 'text', 'vision']
        self.policy.add_to_window(2)  # Index of vision token
        
        self.assertEqual(len(self.policy.vision_window), 1)
        self.assertEqual(self.policy.vision_window[0], 2)
        
        # Add another vision token
        self.policy.token_types.append('vision')
        self.policy.add_to_window(3)
        self.assertEqual(len(self.policy.vision_window), 2)
    
    def test_window_size_limit(self):
        """Test that vision window respects size limit."""
        # Add more tokens than window size
        for i in range(self.vision_window_size + 10):
            self.policy.token_types.append('vision')
            self.policy.add_to_window(i)
        
        # Window should only contain most recent N tokens
        self.assertEqual(len(self.policy.vision_window), self.vision_window_size)
        
        # Should contain the last N indices
        expected_indices = list(range(10, self.vision_window_size + 10))
        self.assertEqual(list(self.policy.vision_window), expected_indices)
    
    def test_eviction_logic(self):
        """Test eviction logic."""
        # Create mock KV cache
        class MockKVCache:
            def __init__(self):
                self.token_metadata = []
            
            def __len__(self):
                return len(self.token_metadata)
        
        mock_cache = MockKVCache()
        
        # Add some tokens
        for i in range(50):
            if i % 10 == 0:
                mock_cache.token_metadata.append(type('obj', (object,), {
                    'is_outlier': True
                }))
            else:
                mock_cache.token_metadata.append(type('obj', (object,), {
                    'is_outlier': False
                }))
        
        self.policy.token_types = ['text'] * 40 + ['vision'] * 10
        
        # Test eviction when adding new tokens
        evict_indices = self.policy(mock_cache, new_tokens=60)
        
        # Should evict some tokens since 50 + 60 > 100
        self.assertGreater(len(evict_indices), 0)
        self.assertLessEqual(len(evict_indices), 10)  # Should evict 10 tokens


class TestKVCache(unittest.TestCase):
    """Test cases for KVCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.max_tokens = 100
        self.dim = 64
        self.num_heads = 4
        self.vision_window_size = 20
        
        self.cache = KVCache(
            max_tokens=self.max_tokens,
            dim=self.dim,
            num_heads=self.num_heads,
            vision_window_size=self.vision_window_size
        )
    
    def test_initialization(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.max_tokens, self.max_tokens)
        self.assertEqual(self.cache.dim, self.dim)
        self.assertEqual(self.cache.num_heads, self.num_heads)
        self.assertEqual(len(self.cache), 0)
    
    def test_add_tokens(self):
        """Test adding tokens to cache."""
        # Add some tokens
        k = torch.randn(2, self.num_heads, 10, self.dim)
        v = torch.randn(2, self.num_heads, 10, self.dim)
        
        k_out, v_out = self.cache.add(k, v, token_type='text')
        
        self.assertEqual(len(self.cache), 10)
        self.assertEqual(k_out.shape, (2, self.num_heads, 10, self.dim))
        self.assertEqual(v_out.shape, (2, self.num_heads, 10, self.dim))
    
    def test_max_tokens_limit(self):
        """Test that cache respects max tokens limit."""
        # Add tokens up to and beyond limit
        for i in range(self.max_tokens + 50):
            k = torch.randn(2, self.num_heads, 1, self.dim)
            v = torch.randn(2, self.num_heads, 1, self.dim)
            token_type = 'vision' if i % 10 == 0 else 'text'
            
            self.cache.add(k, v, token_type=token_type, timestamp=i)
        
        # Cache should not exceed max_tokens
        self.assertLessEqual(len(self.cache), self.max_tokens)
    
    def test_get_tokens(self):
        """Test getting all cached tokens."""
        # Add tokens
        k = torch.randn(2, self.num_heads, 5, self.dim)
        v = torch.randn(2, self.num_heads, 5, self.dim)
        self.cache.add(k, v, token_type='text')
        
        # Get all tokens
        k_all, v_all = self.cache.get()
        
        self.assertEqual(k_all.shape, (2, self.num_heads, 5, self.dim))
        self.assertEqual(v_all.shape, (2, self.num_heads, 5, self.dim))
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        # Add tokens
        k = torch.randn(2, self.num_heads, 10, self.dim)
        v = torch.randn(2, self.num_heads, 10, self.dim)
        self.cache.add(k, v, token_type='text')
        
        memory = self.cache.get_memory_usage()
        
        # Memory should be positive
        self.assertGreater(memory, 0)
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Add tokens
        k = torch.randn(2, self.num_heads, 10, self.dim)
        v = torch.randn(2, self.num_heads, 10, self.dim)
        self.cache.add(k, v, token_type='text')
        
        # Clear cache
        self.cache.clear()
        
        self.assertEqual(len(self.cache), 0)
    
    def test_get_stats(self):
        """Test cache statistics."""
        # Add tokens
        k = torch.randn(2, self.num_heads, 10, self.dim)
        v = torch.randn(2, self.num_heads, 10, self.dim)
        self.cache.add(k, v, token_type='text')
        
        k = torch.randn(2, self.num_heads, 5, self.dim)
        v = torch.randn(2, self.num_heads, 5, self.dim)
        self.cache.add(k, v, token_type='vision')
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['total_tokens'], 15)
        self.assertIn('text', stats['token_types'])
        self.assertIn('vision', stats['token_types'])
        self.assertEqual(stats['token_types']['text'], 10)
        self.assertEqual(stats['token_types']['vision'], 5)
    
    def test_outlier_handling(self):
        """Test handling of outlier tokens."""
        # Add outlier token
        k = torch.randn(2, self.num_heads, 1, self.dim)
        v = torch.randn(2, self.num_heads, 1, self.dim)
        self.cache.add(k, v, token_type='text', is_outlier=True)
        
        # Add non-outlier token
        k = torch.randn(2, self.num_heads, 1, self.dim)
        v = torch.randn(2, self.num_heads, 1, self.dim)
        self.cache.add(k, v, token_type='text', is_outlier=False)
        
        stats = self.cache.get_stats()
        self.assertEqual(stats['outlier_count'], 1)


class TestSlidingWindowKVCache(unittest.TestCase):
    """Test cases for SlidingWindowKVCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.max_tokens = 100
        self.dim = 64
        self.num_heads = 4
        self.vision_window_size = 10
        
        self.cache = SlidingWindowKVCache(
            max_tokens=self.max_tokens,
            dim=self.dim,
            num_heads=self.num_heads,
            vision_window_size=self.vision_window_size
        )
    
    def test_vision_window(self):
        """Test that vision tokens are added to separate window."""
        # Add vision tokens
        for i in range(5):
            k = torch.randn(2, self.num_heads, 1, self.dim)
            v = torch.randn(2, self.num_heads, 1, self.dim)
            self.cache.add(k, v, token_type='vision', timestamp=i)
        
        # Vision window should have 5 tokens
        self.assertEqual(len(self.cache.vision_k_window), 5)
        self.assertEqual(len(self.cache.vision_v_window), 5)
    
    def test_vision_window_size_limit(self):
        """Test that vision window respects size limit."""
        # Add more vision tokens than window size
        for i in range(self.vision_window_size + 5):
            k = torch.randn(2, self.num_heads, 1, self.dim)
            v = torch.randn(2, self.num_heads, 1, self.dim)
            self.cache.add(k, v, token_type='vision', timestamp=i)
        
        # Window should only contain most recent N tokens
        self.assertEqual(len(self.cache.vision_k_window), self.vision_window_size)
    
    def test_mixed_tokens(self):
        """Test handling of mixed token types."""
        # Add text tokens
        for i in range(5):
            k = torch.randn(2, self.num_heads, 1, self.dim)
            v = torch.randn(2, self.num_heads, 1, self.dim)
            self.cache.add(k, v, token_type='text', timestamp=i)
        
        # Add vision tokens
        for i in range(5):
            k = torch.randn(2, self.num_heads, 1, self.dim)
            v = torch.randn(2, self.num_heads, 1, self.dim)
            self.cache.add(k, v, token_type='vision', timestamp=i + 5)
        
        # Check both caches have tokens
        self.assertEqual(len(self.cache.k_cache), 5)  # Text tokens
        self.assertEqual(len(self.cache.vision_k_window), 5)  # Vision tokens
        
        # Total should be 10
        self.assertEqual(len(self.cache), 10)


class TestEvictionPriorities(unittest.TestCase):
    """Test eviction priority logic."""
    
    def test_text_eviction_priority(self):
        """Test that text tokens are evicted before vision tokens."""
        cache = KVCache(
            max_tokens=50,
            dim=64,
            num_heads=4,
            vision_window_size=10
        )
        
        # Add many text tokens
        for i in range(30):
            k = torch.randn(2, 4, 1, 64)
            v = torch.randn(2, 4, 1, 64)
            cache.add(k, v, token_type='text', timestamp=i)
        
        # Add some vision tokens
        for i in range(30):
            k = torch.randn(2, 4, 1, 64)
            v = torch.randn(2, 4, 1, 64)
            cache.add(k, v, token_type='vision', timestamp=i + 30)
        
        stats = cache.get_stats()
        
        # Should have more vision tokens than text tokens
        # (text tokens should have been evicted first)
        text_count = stats['token_types'].get('text', 0)
        vision_count = stats['token_types'].get('vision', 0)
        
        self.assertGreater(vision_count, text_count)


if __name__ == '__main__':
    # Run tests
    print("Running Eviction Unit Tests...")
    print("=" * 50)
    unittest.main()
