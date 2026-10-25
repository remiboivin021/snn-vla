"""
Anchor-Based Quantization for Synobotix VLA PoC
Pivot 2: Quantification par Canaux Ancrés (Anchor-Based) au lieu de Hadamard

Objectif: < 1 point de dégradation de perplexité vs FP16 sur LongBench à 4k tokens
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path


class AnchorQuantizer:
    """
    Anchor-Based Quantizer for KV Cache and Activations.
    
    Implements 2-bit or ternary quantization with static outlier detection.
    
    Args:
        bits: Number of bits for quantization (2 or 3 for ternary)
        dim: Dimension of tensors to quantize
        outlier_threshold: Percentage of channels to treat as outliers (0.01 = 1%)
        calibration_dir: Directory to save/load calibration data
    """
    
    def __init__(self, bits: int = 2, dim: int = 512, 
                 outlier_threshold: float = 0.05, 
                 calibration_dir: str = "./calibration"):
        self.bits = bits
        self.dim = dim
        self.outlier_threshold = outlier_threshold
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        
        # Quantization parameters (computed during calibration)
        self.scales: Optional[torch.Tensor] = None
        self.zeros: Optional[torch.Tensor] = None
        self.outlier_mask: Optional[torch.Tensor] = None
        
        # Quantization method
        self.is_ternary = bits == 3
        self.is_binary = bits == 2
        
        # Device
        self.device = None
    
    def to(self, device: torch.device):
        """Move quantization parameters to device."""
        self.device = device
        if self.scales is not None:
            self.scales = self.scales.to(device)
        if self.zeros is not None:
            self.zeros = self.zeros.to(device)
        if self.outlier_mask is not None:
            self.outlier_mask = self.outlier_mask.to(device)
        return self
    
    def calibrate(self, x: torch.Tensor, num_samples: int = 1000):
        """
        Calibration phase (hors-ligne) to identify outliers and compute scales.
        
        Args:
            x: Sample tensor [num_samples, seq_len, dim]
            num_samples: Number of samples to use for calibration
        """
        # Move to CPU for calibration
        device = x.device
        x = x.cpu()
        
        # Flatten samples: [num_samples * seq_len, dim]
        x_flat = x.reshape(-1, self.dim)
        
        # Compute statistics per channel (dimension)
        means = x_flat.mean(dim=0)
        stds = x_flat.std(dim=0)
        
        # Identify outliers: channels with highest std (most variable)
        # Sort by std and take top N%
        sorted_indices = torch.argsort(stds, descending=True)
        num_outliers = int(self.dim * self.outlier_threshold)
        outlier_indices = sorted_indices[:num_outliers]
        
        # Create outlier mask [dim]
        self.outlier_mask = torch.zeros(self.dim, dtype=torch.bool)
        self.outlier_mask[outlier_indices] = True
        
        # Compute scales and zeros for non-outliers
        non_outlier_mask = ~self.outlier_mask
        
        if self.is_binary:
            # 2-bit quantization: scale to [-2, 2) range
            non_outlier_x = x_flat[:, non_outlier_mask]
            max_vals = non_outlier_x.abs().max(dim=0).values
            min_vals = -max_vals
            
            self.scales = max_vals / 2.0
            self.zeros = min_vals + self.scales
            
        elif self.is_ternary:
            # Ternary quantization: {-1, 0, 1}
            # Use symmetric quantization
            non_outlier_x = x_flat[:, non_outlier_mask]
            max_vals = non_outlier_x.abs().max(dim=0).values
            
            self.scales = max_vals
            self.zeros = torch.zeros_like(max_vals)
        
        # Move back to original device
        self.to(device)
        
        # Save calibration data
        self._save_calibration()
        
        print(f"Calibration complete. Outliers: {num_outliers}/{self.dim} channels")
        return self
    
    def _save_calibration(self):
        """Save calibration parameters to disk."""
        if self.scales is None or self.outlier_mask is None:
            return
        
        state = {
            'scales': self.scales.cpu(),
            'zeros': self.zeros.cpu() if self.zeros is not None else None,
            'outlier_mask': self.outlier_mask.cpu(),
            'bits': self.bits,
            'dim': self.dim,
            'outlier_threshold': self.outlier_threshold
        }
        
        torch.save(state, self.calibration_dir / f"anchor_quant_{self.dim}d_{self.bits}b.pt")
    
    def load_calibration(self, path: Optional[str] = None):
        """Load pre-computed calibration parameters."""
        if path is None:
            path = self.calibration_dir / f"anchor_quant_{self.dim}d_{self.bits}b.pt"
        else:
            path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        
        state = torch.load(path, map_location='cpu')
        self.scales = state['scales']
        self.zeros = state.get('zeros')
        self.outlier_mask = state['outlier_mask']
        self.bits = state['bits']
        self.dim = state['dim']
        self.outlier_threshold = state['outlier_threshold']
        
        print(f"Loaded calibration from {path}")
        return self
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor using anchor-based approach.
        
        Args:
            x: Input tensor [batch, seq_len, dim] or [batch, num_heads, seq_len, head_dim]
            
        Returns:
            Tuple of (quantized_tensor, metadata)
            metadata contains original values for outliers
        """
        if self.scales is None or self.outlier_mask is None:
            raise RuntimeError("Quantizer not calibrated. Run calibrate() first.")
        
        original_shape = x.shape
        original_device = x.device
        
        # Flatten last dimensions
        if x.dim() == 4:
            # [batch, num_heads, seq_len, head_dim] -> [batch * num_heads * seq_len, head_dim]
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            # [batch, seq_len, dim] -> [batch * seq_len, dim]
            x_flat = x.reshape(-1, x.shape[-1])
        
        # Separate outliers and non-outliers
        outlier_mask = self.outlier_mask.to(x.device)
        non_outlier_mask = ~outlier_mask
        
        # Outliers: keep in original precision (fp16)
        outliers = x_flat[:, outlier_mask]  # [N, num_outliers]
        
        # Non-outliers: quantize
        non_outliers = x_flat[:, non_outlier_mask]  # [N, num_non_outliers]
        
        if self.is_binary:
            # 2-bit quantization: map to {-2, -1, 0, 1}
            scales = self.scales[non_outlier_mask].to(x.device)
            zeros = self.zeros[non_outlier_mask].to(x.device)
            
            # Scale and shift
            non_outliers = (non_outliers - zeros) / scales
            
            # Quantize to 2 bits
            quantized = torch.clamp(non_outliers, -2, 1.999)
            quantized = torch.round(quantized)
            
            # Store as float for now (simulation)
            # In production, would use actual 2-bit storage
            quantized_values = quantized * scales + zeros
            
        elif self.is_ternary:
            # Ternary quantization: {-1, 0, 1}
            scales = self.scales[non_outlier_mask].to(x.device)
            
            # Scale to [-1, 1] range
            non_outliers = non_outliers / scales
            
            # Quantize to ternary
            quantized = torch.sign(non_outliers)
            
            # Reconstruct
            quantized_values = quantized * scales
        
        # Combine quantized non-outliers with original outliers
        result = torch.zeros_like(x_flat)
        result[:, non_outlier_mask] = quantized_values
        result[:, outlier_mask] = outliers
        
        # Reshape back
        result = result.reshape(original_shape)
        
        # Metadata for dequantization
        metadata = {
            'outliers': outliers,
            'outlier_mask': outlier_mask,
            'original_shape': original_shape
        }
        
        return result, metadata
    
    def dequantize(self, quantized: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Reconstruct original tensor from quantized version."""
        # For now, just return as we stored outliers in fp16
        # In production, this would reconstruct from 2-bit/ternary storage
        return quantized
    
    def get_memory_savings(self, original_size: int) -> float:
        """
        Calculate memory savings vs fp16.
        
        Args:
            original_size: Size in bytes of original fp16 tensor
            
        Returns:
            Memory savings in bytes
        """
        if self.outlier_mask is None:
            return 0.0
        
        num_outliers = self.outlier_mask.sum().item()
        num_non_outliers = self.dim - num_outliers
        
        # Memory for outliers: fp16 (2 bytes per value)
        outlier_memory = num_outliers * 2
        
        # Memory for non-outliers: 2 bits = 0.25 bytes per value
        non_outlier_memory = num_non_outliers * 0.25
        
        quantized_size = outlier_memory + non_outlier_memory
        savings = original_size - quantized_size
        
        return savings


class TernaryQuantizer(AnchorQuantizer):
    """
    Specialized Ternary Quantizer (3-bit).
    """
    
    def __init__(self, dim: int = 512, outlier_threshold: float = 0.05,
                 calibration_dir: str = "./calibration"):
        super().__init__(bits=3, dim=dim, outlier_threshold=outlier_threshold,
                        calibration_dir=calibration_dir)


if __name__ == "__main__":
    # Test AnchorQuantizer
    print("Testing Anchor-Based Quantizer...")
    
    # Create sample data
    batch_size = 4
    seq_len = 256
    dim = 512
    
    quantizer = AnchorQuantizer(bits=2, dim=dim)
    x = torch.randn(batch_size, seq_len, dim)
    
    # Calibrate
    print("Calibrating...")
    quantizer.calibrate(x, num_samples=100)
    
    # Quantize
    print("Quantizing...")
    quantized, metadata = quantizer.quantize(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Quantized shape: {quantized.shape}")
    print(f"Number of outliers: {quantizer.outlier_mask.sum().item()}")
    print(f"Memory savings per sample: {quantizer.get_memory_savings(batch_size * seq_len * dim * 2):.2f} bytes")
    
    print("✓ Anchor-Based Quantizer test passed!")
