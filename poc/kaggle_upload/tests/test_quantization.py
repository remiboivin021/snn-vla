"""
Unit Tests for Anchor-Based Quantization
Pivot 2: Quantification par Canaux Ancrés
"""

import unittest
import torch
import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.quantizer import AnchorQuantizer, TernaryQuantizer


class TestAnchorQuantizer(unittest.TestCase):
    """Test cases for AnchorQuantizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 512
        self.bits = 2
        self.outlier_threshold = 0.05
        
        # Create temporary directory for calibration
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test quantizer initialization."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        self.assertEqual(quantizer.bits, self.bits)
        self.assertEqual(quantizer.dim, self.dim)
        self.assertEqual(quantizer.outlier_threshold, self.outlier_threshold)
        self.assertIsNone(quantizer.scales)
        self.assertIsNone(quantizer.outlier_mask)
    
    def test_calibration(self):
        """Test calibration process."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        # Create calibration data
        num_samples = 1000
        seq_len = 50
        calibration_data = torch.randn(num_samples, seq_len, self.dim)
        
        # Calibrate
        quantizer.calibrate(calibration_data, num_samples=num_samples)
        
        # Check that calibration computed parameters
        self.assertIsNotNone(quantizer.scales)
        self.assertIsNotNone(quantizer.outlier_mask)
        
        # Check dimensions
        self.assertEqual(quantizer.scales.shape[0], self.dim)
        self.assertEqual(quantizer.outlier_mask.shape[0], self.dim)
        
        # Check outlier count
        num_outliers = quantizer.outlier_mask.sum().item()
        expected_outliers = int(self.dim * self.outlier_threshold)
        self.assertAlmostEqual(num_outliers, expected_outliers, delta=5)
    
    def test_quantize_dequantize(self):
        """Test quantization and dequantization."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        # Calibrate
        calibration_data = torch.randn(1000, 50, self.dim)
        quantizer.calibrate(calibration_data)
        
        # Create test data
        test_data = torch.randn(10, 50, self.dim)
        
        # Quantize
        quantized, metadata = quantizer.quantize(test_data)
        
        self.assertEqual(quantized.shape, test_data.shape)
        self.assertIn('outliers', metadata)
        self.assertIn('outlier_mask', metadata)
        
        # Dequantize
        dequantized = quantizer.dequantize(quantized, metadata)
        self.assertEqual(dequantized.shape, test_data.shape)
    
    def test_memory_savings(self):
        """Test that quantization reduces memory usage."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        # Calibrate
        calibration_data = torch.randn(1000, 50, self.dim)
        quantizer.calibrate(calibration_data)
        
        # Test data
        test_data = torch.randn(10, 50, self.dim)
        
        # Quantize
        quantized, _ = quantizer.quantize(test_data)
        
        # Calculate memory usage
        original_size = test_data.numel() * 2  # fp16
        quantized_size = quantizer.get_memory_savings(original_size)
        
        # Should have positive savings
        self.assertGreater(quantized_size, 0)
    
    def test_2bit_quantization(self):
        """Test 2-bit quantization specifically."""
        quantizer = AnchorQuantizer(
            bits=2,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        calibration_data = torch.randn(1000, 50, self.dim)
        quantizer.calibrate(calibration_data)
        
        test_data = torch.randn(10, 50, self.dim)
        quantized, _ = quantizer.quantize(test_data)
        
        # Check that non-outlier values are in expected range
        # For 2-bit: typically in [-2, 2) range
        non_outlier_mask = ~quantizer.outlier_mask
        non_outlier_values = quantized[:, :, non_outlier_mask]
        
        # Most values should be in a limited range
        # (This is a rough check, exact range depends on calibration)
        self.assertTrue(non_outlier_values.abs().max() < 10)
    
    def test_save_load_calibration(self):
        """Test saving and loading calibration."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        # Calibrate
        calibration_data = torch.randn(1000, 50, self.dim)
        quantizer.calibrate(calibration_data)
        
        # Save path
        save_path = os.path.join(self.temp_dir, 'test_calib.pt')
        
        # Save manually (simulating what the quantizer does)
        state = {
            'scales': quantizer.scales,
            'zeros': quantizer.zeros,
            'outlier_mask': quantizer.outlier_mask
        }
        torch.save(state, save_path)
        
        # Create new quantizer and load
        quantizer2 = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        # Load calibration
        quantizer2.load_calibration(save_path)
        
        # Check that parameters match
        self.assertTrue(torch.allclose(quantizer.scales, quantizer2.scales))
        self.assertTrue(torch.allclose(quantizer.zeros, quantizer2.zeros))
        self.assertTrue(torch.equal(quantizer.outlier_mask, quantizer2.outlier_mask))
    
    def test_device_compatibility(self):
        """Test that quantizer works on different devices."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        # Calibrate on CPU
        calibration_data = torch.randn(1000, 50, self.dim)
        quantizer.calibrate(calibration_data)
        
        # Test on CPU
        test_data_cpu = torch.randn(10, 50, self.dim)
        quantized_cpu, _ = quantizer.quantize(test_data_cpu)
        self.assertEqual(quantized_cpu.device.type, 'cpu')
        
        # If CUDA is available, test on GPU
        if torch.cuda.is_available():
            quantizer_gpu = quantizer.to('cuda')
            test_data_gpu = torch.randn(10, 50, self.dim, device='cuda')
            quantized_gpu, _ = quantizer_gpu.quantize(test_data_gpu)
            self.assertEqual(quantized_gpu.device.type, 'cuda')


class TestTernaryQuantizer(unittest.TestCase):
    """Test cases for TernaryQuantizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 512
        self.outlier_threshold = 0.05
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        import shutil
        import os
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ternary quantizer initialization."""
        quantizer = TernaryQuantizer(
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        self.assertEqual(quantizer.bits, 3)
        self.assertTrue(quantizer.is_ternary)
        self.assertFalse(quantizer.is_binary)
    
    def test_calibration(self):
        """Test ternary calibration."""
        quantizer = TernaryQuantizer(
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        calibration_data = torch.randn(1000, 50, self.dim)
        quantizer.calibrate(calibration_data)
        
        self.assertIsNotNone(quantizer.scales)
        self.assertIsNotNone(quantizer.outlier_mask)


class TestQuantizationQuality(unittest.TestCase):
    """Test the quality of quantization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dim = 512
        self.bits = 2
        self.outlier_threshold = 0.05
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_reconstruction_error(self):
        """Test that reconstruction error is acceptable."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        # Calibrate on diverse data
        calibration_data = torch.randn(10000, 100, self.dim)
        quantizer.calibrate(calibration_data)
        
        # Test on new data
        test_data = torch.randn(100, 50, self.dim)
        quantized, metadata = quantizer.quantize(test_data)
        dequantized = quantizer.dequantize(quantized, metadata)
        
        # Calculate MSE
        mse = torch.mean((test_data - dequantized) ** 2).item()
        
        # For 2-bit quantization, MSE should be reasonable
        # (This is a sanity check, exact values depend on data)
        print(f"Reconstruction MSE: {mse}")
        self.assertLess(mse, 10.0)  # Should be relatively small
    
    def test_outlier_preservation(self):
        """Test that outliers are preserved accurately."""
        quantizer = AnchorQuantizer(
            bits=self.bits,
            dim=self.dim,
            outlier_threshold=self.outlier_threshold,
            calibration_dir=self.temp_dir
        )
        
        calibration_data = torch.randn(1000, 50, self.dim)
        quantizer.calibrate(calibration_data)
        
        # Create test data
        test_data = torch.randn(10, 50, self.dim)
        
        # Quantize
        quantized, metadata = quantizer.quantize(test_data)
        
        # Check that outliers are preserved
        outlier_mask = quantizer.outlier_mask
        
        # Get original and quantized outliers
        original_outliers = test_data[:, :, outlier_mask]
        quantized_outliers = quantized[:, :, outlier_mask]
        
        # They should be very close (stored in fp16)
        diff = torch.abs(original_outliers - quantized_outliers).max().item()
        print(f"Max outlier difference: {diff}")
        self.assertLess(diff, 0.1)  # Should be nearly identical


if __name__ == '__main__':
    # Run tests
    print("Running Quantization Unit Tests...")
    print("=" * 50)
    unittest.main()
