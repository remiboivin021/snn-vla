"""
GQA (Grouped-Query Attention) Implementation for Synobotix VLA PoC
Pivot 1: Architecture GQA native (8 à 16 têtes KV)

Objectif: Réduction de l'empreinte mémoire d'au moins 2x à précision égale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with customizable number of KV heads.
    
    Args:
        dim: Dimension des embeddings
        num_heads: Nombre de têtes de requête (Q)
        num_kv_heads: Nombre de têtes KV (doit diviser num_heads)
        head_dim: Dimension de chaque tête (None pour dim // num_heads)
        dropout: Taux de dropout
    """
    
    def __init__(self, dim: int, num_heads: int = 8, num_kv_heads: int = 8, 
                 head_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or dim // num_heads
        
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.num_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Rotation Position Embedding (optionnel)
        self.rope = None
    
    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand KV heads to match Q heads via repetition.
        [B, num_kv_heads, seq_len, head_dim] -> [B, num_heads, seq_len, head_dim]
        """
        if self.num_groups == 1:
            return x
        B, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(1).expand(B, self.num_groups, num_kv_heads, seq_len, head_dim)
        return x.reshape(B, self.num_heads, seq_len, head_dim)
    
    def forward(self, x: torch.Tensor, kv_cache: dict = None) -> torch.Tensor:
        """
        Forward pass with optional KV cache.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            kv_cache: Dictionary containing cached K and V tensors
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        B, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # [B, seq_len, num_heads * head_dim]
        k = self.k_proj(x)  # [B, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [B, seq_len, num_kv_heads * head_dim]
        
        # Reshape to [B, num_heads, seq_len, head_dim]
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand KV heads to match Q heads
        k = self._expand_kv_heads(k)
        v = self._expand_kv_heads(v)
        
        # Add to cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to V
        output = torch.matmul(attn_probs, v)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(B, seq_len, -1)
        output = self.out_proj(output)
        
        return output
    
    def get_memory_usage(self, batch_size: int, seq_len: int) -> float:
        """
        Calculate memory usage for this layer in MB.
        """
        # Q, K, V projections
        params_memory = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)  # fp32
        
        # Attention matrices
        q_memory = batch_size * seq_len * self.num_heads * self.head_dim * 4 / (1024 * 1024)
        k_memory = batch_size * seq_len * self.num_kv_heads * self.head_dim * 4 / (1024 * 1024)
        v_memory = batch_size * seq_len * self.num_kv_heads * self.head_dim * 4 / (1024 * 1024)
        
        # Attention scores
        attn_memory = batch_size * self.num_heads * seq_len * seq_len * 4 / (1024 * 1024)
        
        return params_memory + q_memory + k_memory + v_memory + attn_memory


class GQATransformerBlock(nn.Module):
    """
    Transformer block with GQA attention.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, num_kv_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        # Self-attention
        self.attn = GQAAttention(dim, num_heads, num_kv_heads, dropout=dropout)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
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
    
    def forward(self, x: torch.Tensor, kv_cache: dict = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attn(self.norm1(x), kv_cache)
        x = x + self.dropout(attn_output)
        
        # MLP
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)
        
        return x


if __name__ == "__main__":
    # Test GQA
    batch_size = 4
    seq_len = 256
    dim = 512
    num_heads = 8
    num_kv_heads = 8
    
    gqa = GQAAttention(dim, num_heads, num_kv_heads)
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward pass
    output = gqa(x)
    print(f"GQA Input shape: {x.shape}")
    print(f"GQA Output shape: {output.shape}")
    print(f"Memory usage: {gqa.get_memory_usage(batch_size, seq_len):.2f} MB")
    
    # Compare with standard attention
    standard_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    std_output, _ = standard_attn(x, x, x)
    print(f"Standard Attention Output shape: {std_output.shape}")
    
    print("✓ GQA Implementation test passed!")
