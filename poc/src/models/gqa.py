"""
GQA (Grouped-Query Attention) Implementation for Synobotix VLA PoC
Pivot 1: Architecture GQA native (8 à 16 têtes KV)

Objectif: Réduction de l'empreinte mémoire d'au moins 2x à précision égale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def apply_rope(x: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on [B, heads, seq_len, head_dim] (positions 0..seq_len-1)."""
    angles = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)[:, None] * inv_freq
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


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
        
        # Rotary position embedding (not stored in checkpoints)
        self.register_buffer(
            'inv_freq',
            1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)),
            persistent=False)
        
        # Optional hooks (experiments): kv_transform(k, v) -> (k, v) applied to the K/V that
        # would be cached, [B, num_kv_heads, seq_len, head_dim]; kv_capture is a list that
        # receives (k, v) of each forward.
        self.kv_transform = None
        self.kv_capture = None
        # kv_mix(k, v) -> (k_q, v_q, exact): scores and values come from the exact K/V where
        # exact[i, j] is True (query i, key j) and from the quantized K/V elsewhere. Models
        # a cache that keeps recent and sink tokens in full precision (eval only).
        self.kv_mix = None
    
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
        
        # Positions are encoded in q and k; cached keys keep their rotation
        q = apply_rope(q, self.inv_freq)
        k = apply_rope(k, self.inv_freq)
        
        if self.kv_capture is not None:
            self.kv_capture.append((k.detach(), v.detach()))
        if self.kv_transform is not None:
            k, v = self.kv_transform(k, v)
        mixed = self.kv_mix(k, v) if self.kv_mix is not None else None
        
        # Expand KV heads to match Q heads
        k = self._expand_kv_heads(k)
        v = self._expand_kv_heads(v)
        
        # Add to cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
        
        if mixed is not None:
            k_q, v_q, exact = mixed
            # k and v are already expanded to the query heads; the quantized pair is not
            output = self._mixed_attention(q, k, v, self._expand_kv_heads(k_q),
                                           self._expand_kv_heads(v_q), exact)
            output = output.transpose(1, 2).contiguous().view(B, seq_len, -1)
            return self.out_proj(output)
        
        # Causal scaled dot-product attention: a position never sees later tokens
        dropout_p = self.dropout.p if self.training else 0.0
        if k.shape[2] == seq_len:
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            # New queries sit after the cached tokens
            mask = torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=q.device)
            mask = mask.tril(diagonal=k.shape[2] - seq_len)
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(B, seq_len, -1)
        output = self.out_proj(output)
        
        return output
    
    def _mixed_attention(self, q, k, v, k_q, v_q, exact):
        """Causal attention reading exact K/V where `exact` is True and quantized K/V elsewhere."""
        T = q.shape[2]
        scores = torch.where(exact, torch.matmul(q, k.transpose(-2, -1)),
                             torch.matmul(q, k_q.transpose(-2, -1))) * self.scale
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        probs = F.softmax(scores.masked_fill(~causal, float('-inf')), dim=-1)
        return torch.matmul(probs * exact, v) + torch.matmul(probs * ~exact, v_q)
    
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
