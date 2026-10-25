"""
KV Cache with Asymmetric Eviction Policy for Synobotix VLA PoC
Pivot 3: Éviction Asymétrique Spatio-Temporelle (Texte vs Vision)

Objectif: Taux de réussite de saisie stable sur tâches de 5 min
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class TokenMetadata:
    """Metadata for each token in the KV cache."""
    token_id: int
    token_type: str  # 'text', 'vision', 'metadata'
    timestamp: int   # Step in sequence
    importance: float = 1.0  # Importance score for eviction
    is_outlier: bool = False  # Whether this token is an outlier


class AsymmetricEvictionPolicy:
    """
    Asymmetric Eviction Policy for KV Cache.
    
    Implements:
    - Sanctuarization of recent vision tokens (high fidelity)
    - Aggressive compression of stable text tokens
    - Sliding window for vision tokens
    
    Args:
        vision_window_size: Number of recent vision tokens to protect
        text_compression_ratio: Ratio of text tokens to compress
        max_tokens: Maximum number of tokens in cache
    """
    
    def __init__(self, vision_window_size: int = 100, 
                 text_compression_ratio: float = 0.25,
                 max_tokens: int = 32768):
        self.vision_window_size = vision_window_size
        self.text_compression_ratio = text_compression_ratio
        self.max_tokens = max_tokens
        
        # Sliding window for vision tokens
        self.vision_window = deque(maxlen=vision_window_size)
        
        # Track token types and positions
        self.token_types = []
        self.token_timestamps = []
    
    def __call__(self, kv_cache: 'KVCache', new_tokens: int) -> List[int]:
        """
        Determine which tokens to evict.
        
        Args:
            kv_cache: The KV cache instance
            new_tokens: Number of new tokens being added
            
        Returns:
            List of token indices to evict
        """
        evict_indices = []
        current_size = len(kv_cache)
        
        if current_size + new_tokens <= self.max_tokens:
            return evict_indices
        
        # Need to evict (current_size + new_tokens - max_tokens) tokens
        num_to_evict = current_size + new_tokens - self.max_tokens
        
        # Strategy: evict text tokens first, then old vision tokens
        # Protect recent vision tokens (in sliding window)
        
        # First, try to evict text tokens that are not outliers
        text_indices = [i for i, t in enumerate(self.token_types) 
                       if t == 'text' and not kv_cache.token_metadata[i].is_outlier]
        
        if len(text_indices) >= num_to_evict:
            # Evict oldest text tokens first
            evict_indices = sorted(text_indices)[:num_to_evict]
        else:
            # Evict all compressible text tokens
            evict_indices = text_indices.copy()
            remaining = num_to_evict - len(evict_indices)
            
            # Then evict old vision tokens (outside sliding window)
            old_vision_indices = [i for i, (t, ts) in enumerate(zip(self.token_types, self.token_timestamps))
                                if t == 'vision' and i not in self.vision_window]
            
            if len(old_vision_indices) >= remaining:
                evict_indices.extend(sorted(old_vision_indices)[:remaining])
            else:
                # As last resort, evict oldest tokens regardless of type
                all_indices = list(range(len(self.token_types)))
                evict_indices.extend(sorted([i for i in all_indices if i not in evict_indices])[:remaining])
        
        return evict_indices
    
    def add_to_window(self, token_idx: int):
        """Add a vision token to the sliding window."""
        if self.token_types[token_idx] == 'vision':
            self.vision_window.append(token_idx)


class KVCache:
    """
    KV Cache with Asymmetric Eviction Policy.
    
    Stores Key and Value tensors for attention mechanism.
    Implements 2-bit/ternary quantization for non-outlier tokens.
    
    Args:
        max_tokens: Maximum number of tokens in cache
        dim: Dimension of key/value tensors
        num_heads: Number of attention heads
        vision_window_size: Size of sliding window for vision tokens
        quantizer: Optional quantizer for KV values
    """
    
    def __init__(self, max_tokens: int = 32768, dim: int = 512,
                 num_heads: int = 8, vision_window_size: int = 100,
                 quantizer: Optional[Any] = None):
        self.max_tokens = max_tokens
        self.dim = dim
        self.num_heads = num_heads
        self.vision_window_size = vision_window_size
        self.quantizer = quantizer
        
        # Storage
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.token_metadata: List[TokenMetadata] = []
        
        # Eviction policy
        self.eviction_policy = AsymmetricEvictionPolicy(
            vision_window_size=vision_window_size,
            max_tokens=max_tokens
        )
        
        # Track sequence length
        self.seq_len = 0
    
    def __len__(self):
        return self.seq_len
    
    def add(self, k: torch.Tensor, v: torch.Tensor, token_type: str,
            timestamp: int = None, importance: float = 1.0,
            is_outlier: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add new key-value pairs to cache.
        
        Args:
            k: Key tensor [batch, num_heads, new_seq_len, dim]
            v: Value tensor [batch, num_heads, new_seq_len, dim]
            token_type: Type of tokens ('text', 'vision', 'metadata')
            timestamp: Optional timestamp
            importance: Importance score
            is_outlier: Whether these are outlier tokens
            
        Returns:
            Tuple of updated (k, v) tensors for attention
        """
        batch_size = k.shape[0]
        new_seq_len = k.shape[2]
        
        if timestamp is None:
            timestamp = self.seq_len
        
        # Check if we need to evict
        if self.seq_len + new_seq_len > self.max_tokens:
            # Determine how many to evict
            num_to_evict = self.seq_len + new_seq_len - self.max_tokens
            evict_indices = self.eviction_policy(self, num_to_evict)
            
            # Remove evicted tokens
            if evict_indices:
                self._remove_tokens(evict_indices)
        
        # Add new tokens
        for i in range(new_seq_len):
            # Extract single token
            k_token = k[:, :, i:i+1, :]  # [batch, num_heads, 1, dim]
            v_token = v[:, :, i:i+1, :]
            
            # Quantize if quantizer is available and not outlier
            if self.quantizer is not None and not is_outlier:
                k_token, _ = self.quantizer.quantize(k_token)
                v_token, _ = self.quantizer.quantize(v_token)
            
            self.k_cache.append(k_token)
            self.v_cache.append(v_token)
            
            metadata = TokenMetadata(
                token_id=self.seq_len + i,
                token_type=token_type,
                timestamp=timestamp + i,
                importance=importance,
                is_outlier=is_outlier
            )
            self.token_metadata.append(metadata)
            
            # Update eviction policy
            if token_type == 'vision':
                self.eviction_policy.token_types.append(token_type)
                self.eviction_policy.token_timestamps.append(timestamp + i)
                self.eviction_policy.add_to_window(self.seq_len + i)
            else:
                self.eviction_policy.token_types.append(token_type)
                self.eviction_policy.token_timestamps.append(timestamp + i)
        
        self.seq_len += new_seq_len
        
        # Concatenate all cached tensors
        k_all = torch.cat(self.k_cache, dim=2)  # [batch, num_heads, seq_len, dim]
        v_all = torch.cat(self.v_cache, dim=2)
        
        return k_all, v_all
    
    def _remove_tokens(self, indices: List[int]):
        """Remove tokens at specified indices."""
        # Sort indices in descending order to avoid index shifting
        indices = sorted(indices, reverse=True)
        
        for idx in indices:
            if idx < len(self.k_cache):
                del self.k_cache[idx]
                del self.v_cache[idx]
                del self.token_metadata[idx]
                
                if idx < len(self.eviction_policy.token_types):
                    del self.eviction_policy.token_types[idx]
                    del self.eviction_policy.token_timestamps[idx]
        
        self.seq_len -= len(indices)
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all cached key-value pairs."""
        if not self.k_cache:
            return torch.empty(0), torch.empty(0)
        
        k_all = torch.cat(self.k_cache, dim=2)
        v_all = torch.cat(self.v_cache, dim=2)
        return k_all, v_all
    
    def get_memory_usage(self) -> float:
        """
        Calculate memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        if not self.k_cache:
            return 0.0
        
        # Get size of one k and v tensor
        k_bytes = self.k_cache[0].numel() * 4  # Assuming fp32 for now
        v_bytes = self.v_cache[0].numel() * 4
        
        total_bytes = len(self.k_cache) * (k_bytes + v_bytes)
        return total_bytes / (1024 * 1024)
    
    def clear(self):
        """Clear the cache."""
        self.k_cache.clear()
        self.v_cache.clear()
        self.token_metadata.clear()
        self.eviction_policy.token_types.clear()
        self.eviction_policy.token_timestamps.clear()
        self.seq_len = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        token_type_counts = {}
        for meta in self.token_metadata:
            token_type_counts[meta.token_type] = token_type_counts.get(meta.token_type, 0) + 1
        
        outlier_count = sum(1 for meta in self.token_metadata if meta.is_outlier)
        
        return {
            'total_tokens': self.seq_len,
            'token_types': token_type_counts,
            'outlier_count': outlier_count,
            'memory_mb': self.get_memory_usage(),
            'capacity_percent': (self.seq_len / self.max_tokens) * 100
        }


class SlidingWindowKVCache(KVCache):
    """
    KV Cache with explicit sliding window for vision tokens.
    
    Maintains a separate high-fidelity window for vision tokens.
    """
    
    def __init__(self, max_tokens: int = 32768, dim: int = 512,
                 num_heads: int = 8, vision_window_size: int = 100,
                 quantizer: Optional[Any] = None):
        super().__init__(max_tokens, dim, num_heads, vision_window_size, quantizer)
        
        # Separate high-fidelity storage for vision window
        self.vision_k_window: List[torch.Tensor] = []
        self.vision_v_window: List[torch.Tensor] = []
        self.vision_metadata: List[TokenMetadata] = []
    
    def add(self, k: torch.Tensor, v: torch.Tensor, token_type: str,
            timestamp: int = None, importance: float = 1.0,
            is_outlier: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override to handle vision window separately."""
        batch_size = k.shape[0]
        new_seq_len = k.shape[2]
        
        if timestamp is None:
            timestamp = self.seq_len
        
        # Handle vision tokens separately
        if token_type == 'vision':
            for i in range(new_seq_len):
                k_token = k[:, :, i:i+1, :]
                v_token = v[:, :, i:i+1, :]
                
                # Vision tokens are ALWAYS high fidelity (no quantization)
                self.vision_k_window.append(k_token)
                self.vision_v_window.append(v_token)
                
                metadata = TokenMetadata(
                    token_id=self.seq_len + i,
                    token_type=token_type,
                    timestamp=timestamp + i,
                    importance=importance,
                    is_outlier=is_outlier
                )
                self.vision_metadata.append(metadata)
                
                # Maintain window size
                if len(self.vision_k_window) > self.vision_window_size:
                    del self.vision_k_window[0]
                    del self.vision_v_window[0]
                    del self.vision_metadata[0]
                
                # Also add to main cache for compatibility
                super().add(k_token, v_token, token_type, timestamp + i, importance, is_outlier)
        else:
            # For non-vision tokens, use standard add
            return super().add(k, v, token_type, timestamp, importance, is_outlier)
        
        self.seq_len += new_seq_len
        
        # Combine main cache with vision window
        k_all = torch.cat(self.k_cache + self.vision_k_window, dim=2)
        v_all = torch.cat(self.v_cache + self.vision_v_window, dim=2)
        
        return k_all, v_all


if __name__ == "__main__":
    print("Testing KV Cache with Asymmetric Eviction...")
    
    # Create cache
    cache = KVCache(max_tokens=100, dim=64, num_heads=4, vision_window_size=20)
    
    # Add some tokens
    batch_size = 2
    num_heads = 4
    dim = 64
    
    # Add text tokens
    for i in range(50):
        k = torch.randn(batch_size, num_heads, 1, dim)
        v = torch.randn(batch_size, num_heads, 1, dim)
        cache.add(k, v, token_type='text', timestamp=i)
    
    print(f"After adding text: {len(cache)} tokens")
    print(f"Stats: {cache.get_stats()}")
    
    # Add vision tokens
    for i in range(60):
        k = torch.randn(batch_size, num_heads, 1, dim)
        v = torch.randn(batch_size, num_heads, 1, dim)
        cache.add(k, v, token_type='vision', timestamp=50 + i)
    
    print(f"After adding vision: {len(cache)} tokens")
    print(f"Stats: {cache.get_stats()}")
    print(f"Memory usage: {cache.get_memory_usage():.2f} MB")
    
    print("✓ KV Cache test passed!")
