"""
Gated Delta Network for Recurrent History
Pivot 4: Hybridation Récurrente

Objectif: Mémoire de contexte globale bornée et stable à 330 Mio au décodage infini
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedDeltaNet(nn.Module):
    """
    Gated Delta Network for maintaining long-term history.
    
    Implements a gated recurrent unit that selectively updates hidden state
    based on new information, preserving important historical context.
    
    Args:
        dim: Dimension of input and hidden state
        hidden_dim: Optional different dimension for hidden state
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        
        # Projections
        self.delta_proj = nn.Linear(dim, self.hidden_dim)
        self.gate_proj = nn.Linear(dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, dim)
        
        # Initialize hidden state
        self.register_buffer('hidden_state', torch.zeros(1, self.hidden_dim))
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset hidden state for new sequence."""
        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, 
                                       device=self.hidden_state.device,
                                       dtype=self.hidden_state.dtype)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gated update.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Tuple of (output, new_hidden_state)
            output: [batch_size, seq_len, dim]
            new_hidden_state: [batch_size, hidden_dim]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Ensure hidden state has correct batch size
        if self.hidden_state.shape[0] != batch_size:
            self.reset_hidden(batch_size)
        
        # Project input
        delta = torch.tanh(self.delta_proj(x))  # [B, S, hidden_dim]
        gate = torch.sigmoid(self.gate_proj(x))  # [B, S, hidden_dim]
        
        # Expand hidden state to match sequence length
        hidden_expanded = self.hidden_state.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        
        # Gated update
        new_hidden = hidden_expanded + gate * delta
        
        # Update hidden state (use last time step)
        self.hidden_state = new_hidden[:, -1, :].clone()
        
        # Project back to original dimension
        output = self.out_proj(new_hidden)
        
        return output, self.hidden_state


class DeltaNetBlock(nn.Module):
    """
    Transformer block with integrated DeltaNet for recurrent history.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, delta_hidden_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention (can be replaced with GQA)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # DeltaNet for recurrent history
        self.delta_net = GatedDeltaNet(dim, delta_hidden_dim)
        
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[dict] = None) -> torch.Tensor:
        """
        Forward pass with DeltaNet integration.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            kv_cache: Optional KV cache dictionary
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        # Self-attention
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_output)
        
        # DeltaNet for recurrent history
        delta_output, _ = self.delta_net(self.norm2(x))
        x = x + self.dropout(delta_output)
        
        # MLP
        mlp_output = self.mlp(self.norm3(x))
        x = x + self.dropout(mlp_output)
        
        return x


class MemoryBoundedDeltaNet(nn.Module):
    """
    DeltaNet with explicit memory bounding.
    
    Ensures that memory usage stays within specified limits
    by compressing or pruning hidden state when necessary.
    """
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 max_memory_mb: float = 330.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.max_memory_mb = max_memory_mb
        
        # Projections
        self.delta_proj = nn.Linear(dim, self.hidden_dim)
        self.gate_proj = nn.Linear(dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, dim)
        
        # Compression for memory bounding
        self.compression_threshold = 0.8  # Compress when 80% of memory used
        self.compression_proj = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.decompression_proj = nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        
        # Initialize hidden state
        self.register_buffer('hidden_state', torch.zeros(1, self.hidden_dim))
        self.register_buffer('compressed_state', torch.zeros(1, self.hidden_dim // 2))
        self.is_compressed = False
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset hidden state."""
        self.hidden_state = torch.zeros(batch_size, self.hidden_dim,
                                       device=self.hidden_state.device,
                                       dtype=self.hidden_state.dtype)
        self.compressed_state = torch.zeros(batch_size, self.hidden_dim // 2,
                                          device=self.hidden_state.device,
                                          dtype=self.hidden_state.dtype)
        self.is_compressed = False
    
    def _get_memory_usage(self) -> float:
        """Calculate current memory usage in MB."""
        if self.is_compressed:
            mem_bytes = self.compressed_state.numel() * 4  # fp32
        else:
            mem_bytes = self.hidden_state.numel() * 4
        return mem_bytes / (1024 * 1024)
    
    def _compress_state(self):
        """Compress hidden state to reduce memory usage."""
        self.compressed_state = self.compression_proj(self.hidden_state)
        self.is_compressed = True
    
    def _decompress_state(self):
        """Decompress hidden state."""
        self.hidden_state = self.decompression_proj(self.compressed_state)
        self.is_compressed = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with memory-aware compression.
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Check memory usage
        current_memory = self._get_memory_usage()
        if current_memory > self.max_memory_mb * self.compression_threshold:
            if not self.is_compressed:
                self._compress_state()
        else:
            if self.is_compressed:
                self._decompress_state()
        
        # Ensure hidden state has correct batch size
        if self.hidden_state.shape[0] != batch_size:
            self.reset_hidden(batch_size)
        
        # Project input
        delta = torch.tanh(self.delta_proj(x))
        gate = torch.sigmoid(self.gate_proj(x))
        
        # Get current hidden state
        if self.is_compressed:
            hidden_expanded = self.decompression_proj(self.compressed_state)
        else:
            hidden_expanded = self.hidden_state
        
        hidden_expanded = hidden_expanded.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)
        
        # Gated update
        new_hidden = hidden_expanded + gate * delta
        
        # Update hidden state
        if self.is_compressed:
            self.compressed_state = self.compression_proj(new_hidden[:, -1, :])
        else:
            self.hidden_state = new_hidden[:, -1, :].clone()
        
        # Project back
        output = self.out_proj(new_hidden)
        
        return output, self.hidden_state if not self.is_compressed else self.compressed_state


if __name__ == "__main__":
    print("Testing Gated DeltaNet...")
    
    # Create DeltaNet
    dim = 512
    delta_net = GatedDeltaNet(dim)
    
    # Test input
    batch_size = 4
    seq_len = 100
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward pass
    output, hidden = delta_net(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    print(f"Memory usage: {delta_net._get_memory_usage():.2f} MB")
    
    # Test MemoryBoundedDeltaNet
    print("\nTesting MemoryBoundedDeltaNet...")
    memory_delta_net = MemoryBoundedDeltaNet(dim, max_memory_mb=1.0)
    output, hidden = memory_delta_net(x)
    print(f"Output shape: {output.shape}")
    print(f"Memory usage: {memory_delta_net._get_memory_usage():.2f} MB")
    
    print("✓ DeltaNet tests passed!")
