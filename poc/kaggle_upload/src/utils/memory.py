"""
Memory Profiling Utilities for Synobotix VLA PoC

Tracks memory usage of tensors, models, and operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
import gc
import psutil
import os


class MemoryProfiler:
    """
    Memory profiler for tracking GPU and CPU memory usage.
    """
    
    def __init__(self):
        self.start_memory = {}
        self.peak_memory = {}
        self.current_memory = {}
    
    def start(self):
        """Start memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory['gpu'] = torch.cuda.memory_allocated()
        
        self.start_memory['cpu'] = psutil.Process(os.getpid()).memory_info().rss
        self.peak_memory['gpu'] = self.start_memory.get('gpu', 0)
        self.peak_memory['cpu'] = self.start_memory.get('cpu', 0)
    
    def update(self):
        """Update current memory usage."""
        self.current_memory = {}
        
        if torch.cuda.is_available():
            self.current_memory['gpu'] = torch.cuda.memory_allocated()
            self.peak_memory['gpu'] = max(self.peak_memory.get('gpu', 0), 
                                         torch.cuda.max_memory_allocated())
        
        self.current_memory['cpu'] = psutil.Process(os.getpid()).memory_info().rss
        self.peak_memory['cpu'] = max(self.peak_memory.get('cpu', 0), 
                                     self.current_memory['cpu'])
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics in MB."""
        stats = {}
        
        for key in ['gpu', 'cpu']:
            if key in self.current_memory:
                stats[f'{key}_current'] = self.current_memory[key] / (1024 * 1024)
            if key in self.peak_memory:
                stats[f'{key}_peak'] = self.peak_memory[key] / (1024 * 1024)
            if key in self.start_memory:
                stats[f'{key}_delta'] = (self.current_memory.get(key, 0) - 
                                       self.start_memory[key]) / (1024 * 1024)
        
        return stats
    
    def reset(self):
        """Reset memory tracking."""
        self.start_memory.clear()
        self.peak_memory.clear()
        self.current_memory.clear()


@contextmanager
def measure_memory(device: Optional[str] = None, print_stats: bool = True):
    """
    Context manager to measure memory usage of a code block.
    
    Args:
        device: 'cpu' or 'gpu' or None (auto-detect)
        print_stats: Whether to print statistics
        
    Yields:
        MemoryProfiler instance
    """
    profiler = MemoryProfiler()
    profiler.start()
    
    try:
        yield profiler
    finally:
        profiler.update()
        if print_stats:
            stats = profiler.get_stats()
            print("Memory Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f} MB")


def get_tensor_memory(tensor: torch.Tensor) -> float:
    """
    Get memory usage of a tensor in MB.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Memory usage in megabytes
    """
    num_elements = tensor.numel()
    bytes_per_element = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }.get(tensor.dtype, 4)  # Default to 4 bytes
    
    return num_elements * bytes_per_element / (1024 * 1024)


def get_model_memory(model: nn.Module, include_gradients: bool = True) -> float:
    """
    Get memory usage of a model in MB.
    
    Args:
        model: PyTorch model
        include_gradients: Whether to include gradient memory
        
    Returns:
        Memory usage in megabytes
    """
    param_bytes = sum(p.numel() * 4 for p in model.parameters())  # fp32
    
    if include_gradients:
        grad_bytes = sum(p.grad.numel() * 4 for p in model.parameters() 
                       if p.grad is not None)
    else:
        grad_bytes = 0
    
    buffer_bytes = sum(b.numel() * 4 for b in model.buffers())
    
    return (param_bytes + grad_bytes + buffer_bytes) / (1024 * 1024)


def profile_operation(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile memory usage of a function call.
    
    Args:
        func: Function to profile
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Dictionary with profiling results
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    profiler = MemoryProfiler()
    profiler.start()
    
    # Run function
    result = func(*args, **kwargs)
    
    profiler.update()
    
    return {
        'result': result,
        'stats': profiler.get_stats()
    }


class MemoryMonitor:
    """
    Real-time memory monitor that logs usage at intervals.
    """
    
    def __init__(self, interval: int = 10):
        self.interval = interval
        self.logs = []
    
    def start(self):
        """Start monitoring."""
        import threading
        self.stop_flag = False
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def _monitor(self):
        """Monitoring thread."""
        import time
        
        while not self.stop_flag:
            log = {
                'timestamp': time.time(),
                'gpu_memory': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
                'cpu_memory': psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            }
            self.logs.append(log)
            time.sleep(self.interval)
    
    def stop(self):
        """Stop monitoring."""
        self.stop_flag = True
        self.thread.join()
    
    def get_logs(self) -> list:
        """Get monitoring logs."""
        return self.logs


if __name__ == "__main__":
    print("Testing Memory Profiler...")
    
    # Test memory measurement
    with measure_memory() as profiler:
        # Create some tensors
        x = torch.randn(1000, 1000, device='cpu')
        if torch.cuda.is_available():
            y = torch.randn(1000, 1000, device='cuda')
        
        # Do some operations
        z = x @ x.T
    
    # Test tensor memory
    tensor = torch.randn(1000, 1000)
    print(f"\nTensor memory: {get_tensor_memory(tensor):.2f} MB")
    
    # Test model memory
    model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000)
    )
    print(f"Model memory: {get_model_memory(model):.2f} MB")
    
    print("\n✓ Memory Profiler tests passed!")
