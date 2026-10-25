"""
Metrics Computation Utilities for Synobotix VLA PoC

Computes perplexity, accuracy, and other evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any
from collections import defaultdict
import numpy as np


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor,
                      epsilon: float = 1e-10) -> float:
    """
    Compute perplexity for a batch of predictions.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        targets: Ground truth tokens [batch_size, seq_len]; -1 marks padding and is ignored
        epsilon: Small constant for numerical stability
        
    Returns:
        Perplexity (float)
    """
    # Reshape for cross-entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean', ignore_index=-1)
    
    # Perplexity = exp(ce_loss)
    perplexity = torch.exp(ce_loss).item()
    
    return perplexity


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                    top_k: int = 1) -> float:
    """
    Compute accuracy for a batch of predictions.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        targets: Ground truth tokens [batch_size, seq_len]; -1 marks padding and is ignored
        top_k: Number of top predictions to consider
        
    Returns:
        Accuracy (float)
    """
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # Drop padding positions
    valid = targets_flat != -1
    logits_flat, targets_flat = logits_flat[valid], targets_flat[valid]
    if targets_flat.numel() == 0:
        return 0.0
    
    # Get top-k predictions
    if top_k == 1:
        preds = torch.argmax(logits_flat, dim=-1)
        correct = (preds == targets_flat).float().mean().item()
    else:
        # Top-k accuracy
        top_k_preds = torch.topk(logits_flat, k=top_k, dim=-1).indices
        correct = torch.zeros_like(targets_flat, dtype=torch.float)
        
        for i in range(targets_flat.numel()):
            if targets_flat[i] in top_k_preds[i]:
                correct[i] = 1.0
        
        correct = correct.mean().item()
    
    return correct


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                          token_type: Optional[str] = None,
                          mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute accuracy for specific token types.
    
    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        targets: Ground truth tokens [batch_size, seq_len]
        token_type: Optional token type filter ('text', 'vision', etc.)
        mask: Optional mask tensor [batch_size, seq_len]
        
    Returns:
        Accuracy for the specified token type
    """
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # Get predictions
    preds = torch.argmax(logits_flat, dim=-1)
    
    # Create mask for all tokens
    if mask is not None:
        mask_flat = mask.view(-1)
    else:
        mask_flat = torch.ones_like(targets_flat, dtype=torch.bool)
    
    # Filter by token type if specified
    # (In practice, you would need token type information)
    
    # Compute accuracy
    correct = (preds == targets_flat).float()
    masked_correct = correct[mask_flat]
    
    if masked_correct.numel() == 0:
        return 0.0
    
    return masked_correct.mean().item()


class MetricsTracker:
    """
    Tracks multiple metrics across training/evaluation.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step = 0
    
    def add(self, name: str, value: float):
        """Add a metric value."""
        self.metrics[name].append((self.step, value))
    
    def increment_step(self):
        """Increment the current step."""
        self.step += 1
    
    def get_latest(self, name: str) -> float:
        """Get the latest value for a metric."""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1][1]
        return 0.0
    
    def get_mean(self, name: str, window: int = None) -> float:
        """
        Get the mean value for a metric over a window.
        
        Args:
            name: Metric name
            window: Number of recent steps to average (None for all)
            
        Returns:
            Mean value
        """
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = [v for _, v in self.metrics[name]]
        
        if window is not None:
            values = values[-window:]
        
        return np.mean(values)
    
    def get_all(self) -> Dict[str, float]:
        """Get all latest metric values."""
        return {name: self.get_latest(name) for name in self.metrics}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.step = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {}
        for name, values in self.metrics.items():
            result[name] = {
                'latest': self.get_latest(name),
                'mean': self.get_mean(name),
                'count': len(values)
            }
        return result


class SequenceMetrics:
    """
    Metrics specific to sequence generation tasks.
    """
    
    def __init__(self):
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_sequences = 0
        self.completed_sequences = 0
    
    def update(self, generated: torch.Tensor, targets: torch.Tensor,
              sequence_completed: bool = True):
        """
        Update metrics with new predictions.
        
        Args:
            generated: Generated tokens [batch_size, seq_len]
            targets: Target tokens [batch_size, seq_len]
            sequence_completed: Whether the sequence was completed successfully
        """
        batch_size, seq_len = generated.shape
        
        self.total_tokens += batch_size * seq_len
        self.total_sequences += batch_size
        
        if sequence_completed:
            self.completed_sequences += batch_size
        
        # Token accuracy
        correct = (generated == targets).float().sum().item()
        self.correct_tokens += correct
    
    def get_accuracy(self) -> float:
        """Get token accuracy."""
        if self.total_tokens == 0:
            return 0.0
        return self.correct_tokens / self.total_tokens
    
    def get_completion_rate(self) -> float:
        """Get sequence completion rate."""
        if self.total_sequences == 0:
            return 0.0
        return self.completed_sequences / self.total_sequences
    
    def reset(self):
        """Reset all metrics."""
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_sequences = 0
        self.completed_sequences = 0


class RoboticTaskMetrics:
    """
    Metrics specific to robotic manipulation tasks.
    
    Tracks success rates, precision, and completion times.
    """
    
    def __init__(self):
        self.task_results = []
    
    def add_result(self, task_name: str, success: bool, 
                   completion_time: float, precision: float = 1.0):
        """
        Add a task result.
        
        Args:
            task_name: Name of the task
            success: Whether the task was successful
            completion_time: Time to complete (seconds)
            precision: Precision metric (0-1)
        """
        self.task_results.append({
            'task_name': task_name,
            'success': success,
            'completion_time': completion_time,
            'precision': precision
        })
    
    def get_success_rate(self, task_name: Optional[str] = None) -> float:
        """Get success rate for a specific task or overall."""
        if task_name is not None:
            results = [r for r in self.task_results if r['task_name'] == task_name]
        else:
            results = self.task_results
        
        if not results:
            return 0.0
        
        return sum(1 for r in results if r['success']) / len(results)
    
    def get_avg_completion_time(self, task_name: Optional[str] = None) -> float:
        """Get average completion time."""
        if task_name is not None:
            results = [r for r in self.task_results if r['task_name'] == task_name]
        else:
            results = self.task_results
        
        if not results:
            return 0.0
        
        return sum(r['completion_time'] for r in results) / len(results)
    
    def get_avg_precision(self, task_name: Optional[str] = None) -> float:
        """Get average precision."""
        if task_name is not None:
            results = [r for r in self.task_results if r['task_name'] == task_name]
        else:
            results = self.task_results
        
        if not results:
            return 0.0
        
        return sum(r['precision'] for r in results) / len(results)
    
    def reset(self):
        """Reset all metrics."""
        self.task_results = []


if __name__ == "__main__":
    print("Testing Metrics Utilities...")
    
    # Test perplexity
    logits = torch.randn(4, 10, 50000)
    targets = torch.randint(0, 50000, (4, 10))
    ppl = compute_perplexity(logits, targets)
    print(f"Perplexity: {ppl:.2f}")
    
    # Test accuracy
    acc = compute_accuracy(logits, targets, top_k=1)
    print(f"Accuracy: {acc:.4f}")
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.add('loss', 0.5)
    tracker.add('loss', 0.3)
    tracker.add('accuracy', 0.9)
    tracker.increment_step()
    
    print(f"Latest loss: {tracker.get_latest('loss'):.4f}")
    print(f"Mean loss: {tracker.get_mean('loss'):.4f}")
    print(f"All metrics: {tracker.to_dict()}")
    
    # Test SequenceMetrics
    seq_metrics = SequenceMetrics()
    seq_metrics.update(torch.randint(0, 100, (2, 10)), 
                      torch.randint(0, 100, (2, 10)), 
                      sequence_completed=True)
    print(f"Sequence accuracy: {seq_metrics.get_accuracy():.4f}")
    
    print("\n✓ Metrics tests passed!")
