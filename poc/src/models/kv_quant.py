"""
KIVI / Kitty style KV-cache quantization, to compare our anchor-based scheme with
published methods on the same model and data.

Re-implemented from the Kitty paper's description (arXiv 2511.18643, section 3 and
Algorithm 1), not from its code:
- keys: per channel, groups of `group_size` tokens, statistics computed on the group
- the `boost_ratio` fraction of key channels with the largest mean |x| in each group
  gets 4 bits instead of 2 (Kitty); boost_ratio=0 is KIVI
- values: per token
- the first `sink_tokens` tokens stay in full precision (Kitty, KIVI-K2V2*)
- tokens of the group still being filled stay in full precision (residual buffer)

The paper does not give the scale / zero-point formula: plain min/max uniform
quantization is used here. Quantization is applied to K/V as they are after RoPE.
"""

import torch


def quantize_uniform(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Min/max uniform quantization along `dim`, returned dequantized."""
    levels = 2 ** bits - 1
    low = x.amin(dim=dim, keepdim=True)
    scale = ((x.amax(dim=dim, keepdim=True) - low) / levels).clamp(min=1e-8)
    return torch.round((x - low) / scale) * scale + low


class GroupedKVQuantizer:
    """
    Args:
        group_size: Tokens per key group (and size of the full-precision residual buffer)
        boost_ratio: Fraction of key channels given `boost_bits` (0 = KIVI)
        sink_tokens: Leading tokens kept in full precision
        bits: Base precision of keys and values
        boost_bits: Precision of boosted key channels
    """

    def __init__(self, group_size: int = 128, boost_ratio: float = 0.125, sink_tokens: int = 32,
                 bits: int = 2, boost_bits: int = 4):
        self.group_size = group_size
        self.boost_ratio = boost_ratio
        self.sink_tokens = sink_tokens
        self.bits = bits
        self.boost_bits = boost_bits

    def quantize_keys(self, k: torch.Tensor) -> torch.Tensor:
        B, H, T, D = k.shape
        G = self.group_size
        full = T // G * G
        if full == 0:
            return k
        groups = k[:, :, :full].reshape(B, H, full // G, G, D)
        quantized = quantize_uniform(groups, self.bits, dim=3)
        num_boost = round(self.boost_ratio * D)
        if num_boost > 0:
            boosted = quantize_uniform(groups, self.boost_bits, dim=3)
            top = groups.abs().mean(dim=3).topk(num_boost, dim=-1).indices
            is_boosted = torch.zeros(B, H, full // G, D, dtype=torch.bool, device=k.device)
            is_boosted.scatter_(-1, top, True)
            quantized = torch.where(is_boosted.unsqueeze(3), boosted, quantized)
        return torch.cat([quantized.reshape(B, H, full, D), k[:, :, full:]], dim=2)

    def quantize_values(self, v: torch.Tensor) -> torch.Tensor:
        full = v.shape[2] // self.group_size * self.group_size
        return torch.cat([quantize_uniform(v[:, :, :full], self.bits, dim=-1), v[:, :, full:]], dim=2)

    def exact_mask(self, T: int, device) -> torch.Tensor:
        """exact[i, j]: query i reads key/value j in full precision."""
        group = torch.arange(T, device=device) // self.group_size
        exact = group[:, None] == group[None, :]  # group being filled: residual buffer
        exact[:, :self.sink_tokens] = True
        return exact

    def __call__(self, k: torch.Tensor, v: torch.Tensor):
        """K/V [B, heads, T, head_dim] -> (k_q, v_q, exact) for GQAAttention.kv_mix."""
        return self.quantize_keys(k), self.quantize_values(v), self.exact_mask(k.shape[2], k.device)

    def bytes_for_tokens(self, num_tokens: int, kv_heads: int, head_dim: int) -> float:
        """Bytes one layer's cache needs for `num_tokens`, counting scales and zero-points (fp16)."""
        exact_tokens = min(num_tokens, self.sink_tokens + self.group_size)
        quantized_tokens = num_tokens - exact_tokens
        key_bits = self.bits + self.boost_ratio * (self.boost_bits - self.bits) + 32 / self.group_size
        value_bits = self.bits + 32 / head_dim
        quantized = quantized_tokens * kv_heads * head_dim * (key_bits + value_bits) / 8
        exact = exact_tokens * 2 * kv_heads * head_dim * 2
        return quantized + exact
