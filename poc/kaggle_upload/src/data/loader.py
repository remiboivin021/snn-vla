"""
Data Loader Utilities for Synobotix VLA PoC

Provides DataLoaders for training, evaluation, and calibration.
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml

from .synthetic import SyntheticDataset, RoboticSequenceDataset, CalibrationDataset


def get_dataloader(dataset_type: str = 'synthetic', 
                  batch_size: int = 32, 
                  num_workers: int = 4,
                  seq_length: int = 100,
                  vocab_size: int = 50000,
                  split: str = 'train',
                  **kwargs) -> DataLoader:
    """
    Get a DataLoader for the specified dataset type.
    
    Args:
        dataset_type: 'synthetic', 'robotic', or 'calibration'
        batch_size: Batch size
        num_workers: Number of workers
        seq_length: Sequence length
        vocab_size: Vocabulary size
        split: 'train', 'val', or 'test'
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader instance
    """
    
    if dataset_type == 'synthetic':
        dataset = SyntheticDataset(
            num_samples=10000,
            seq_length=seq_length,
            vocab_size=vocab_size,
            **kwargs
        )
        
        # For synthetic, use the built-in collate function
        collate_fn = dataset.get_collate_fn()
        
    elif dataset_type == 'robotic':
        dataset = RoboticSequenceDataset(
            num_sequences=1000,
            seq_length=seq_length,
            **kwargs
        )
        
        # Custom collate for robotic sequences
        collate_fn = robotic_collate_fn
        
    elif dataset_type == 'calibration':
        dim = kwargs.get('dim', 512)
        dataset = CalibrationDataset(
            num_samples=kwargs.get('num_samples', 10000),
            dim=dim
        )
        collate_fn = default_collate
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Split dataset if needed
    if split != 'train':
        # Create a smaller validation set
        val_size = min(1000, len(dataset) // 10)
        test_size = min(1000, len(dataset) // 10)
        
        if split == 'val':
            dataset, _ = random_split(dataset, [val_size, len(dataset) - val_size])
        elif split == 'test':
            dataset, _ = random_split(dataset, [test_size, len(dataset) - test_size])
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


def get_calibration_loader(batch_size: int = 100, dim: int = 512,
                           num_samples: int = 10000) -> DataLoader:
    """
    Get a DataLoader for calibration.
    
    Args:
        batch_size: Batch size
        dim: Dimension of samples
        num_samples: Number of calibration samples
        
    Returns:
        DataLoader with calibration samples
    """
    dataset = CalibrationDataset(num_samples=num_samples, dim=dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def default_collate(batch: List[torch.Tensor]) -> torch.Tensor:
    """Default collate function for tensors."""
    return torch.stack(batch)


def robotic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for robotic sequences.
    
    Handles mixed token types (text, vision, metadata).
    """
    # Find maximum sequence length
    max_len = max(len(sample['tokens']) for sample in batch)
    
    # Stack tokens
    tokens_list = []
    for sample in batch:
        # Convert all tokens to tensors of same size
        token_tensors = []
        for token in sample['tokens']:
            if isinstance(token, torch.Tensor):
                # Vision or metadata token
                token_tensors.append(token)
            else:
                # Text token (int)
                token_tensors.append(torch.tensor([token], dtype=torch.long))
        
        # Pad sequence
        padded = torch.zeros(max_len, dtype=torch.float32)
        for i, t in enumerate(token_tensors):
            if i < max_len:
                padded[i] = t.float() if t.numel() == 1 else t.mean().float()
        tokens_list.append(padded)
    
    tokens = torch.stack(tokens_list)  # [batch, seq_len]
    
    # Stack targets
    targets_list = []
    for sample in batch:
        target_tensors = []
        for target in sample['targets']:
            if isinstance(target, torch.Tensor):
                target_tensors.append(target.float())
            else:
                target_tensors.append(torch.tensor([target], dtype=torch.float32))
        
        padded = torch.zeros(max_len, dtype=torch.float32)
        for i, t in enumerate(target_tensors):
            if i < max_len:
                padded[i] = t.float() if t.numel() == 1 else t.mean().float()
        targets_list.append(padded)
    
    targets = torch.stack(targets_list)
    
    # Stack token types
    all_token_types = []
    for sample in batch:
        all_token_types.extend(sample['token_types'][:max_len])
    
    # Stack tasks
    tasks = [sample['task'] for sample in batch]
    
    return {
        'input_tokens': tokens,
        'target_tokens': targets,
        'token_types': all_token_types,
        'tasks': tasks
    }


def get_robotic_dataloader(batch_size: int = 32, seq_length: int = 200,
                          split: str = 'train', **kwargs) -> DataLoader:
    """
    Get a DataLoader specifically for robotic sequences.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        split: 'train', 'val', or 'test'
        **kwargs: Additional arguments
        
    Returns:
        DataLoader for robotic sequences
    """
    dataset = RoboticSequenceDataset(
        num_sequences=5000 if split == 'train' else 1000,
        seq_length=seq_length,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        collate_fn=robotic_collate_fn,
        pin_memory=True
    )
    
    return dataloader


def create_data_config(config_path: str = "config/data.yaml") -> Dict[str, Any]:
    """
    Create a default data configuration.
    
    Args:
        config_path: Path to save configuration
        
    Returns:
        Data configuration dictionary
    """
    config = {
        'train': {
            'dataset_type': 'synthetic',
            'batch_size': 32,
            'num_workers': 4,
            'seq_length': 100,
            'vocab_size': 50000
        },
        'val': {
            'dataset_type': 'synthetic',
            'batch_size': 32,
            'num_workers': 2,
            'seq_length': 100,
            'vocab_size': 50000
        },
        'calibration': {
            'batch_size': 100,
            'dim': 512,
            'num_samples': 10000
        },
        'robotic': {
            'batch_size': 16,
            'seq_length': 200,
            'num_actions': 50,
            'observation_dim': 64
        }
    }
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config


def load_data_config(config_path: str = "config/data.yaml") -> Dict[str, Any]:
    """Load data configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    print("Testing Data Loaders...")
    
    # Test synthetic dataloader
    print("\n1. Testing Synthetic DataLoader:")
    synth_loader = get_dataloader('synthetic', batch_size=4, seq_length=50)
    batch = next(iter(synth_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Text tokens shape: {batch['text_tokens'].shape}")
    print(f"Vision patches shape: {batch['vision_patches'].shape}")
    
    # Test calibration dataloader
    print("\n2. Testing Calibration DataLoader:")
    calib_loader = get_calibration_loader(batch_size=10, dim=512)
    calib_batch = next(iter(calib_loader))
    print(f"Calibration batch shape: {calib_batch.shape}")
    
    # Test robotic dataloader
    print("\n3. Testing Robotic DataLoader:")
    robot_loader = get_robotic_dataloader(batch_size=2, seq_length=30)
    robot_batch = next(iter(robot_loader))
    print(f"Robotic batch keys: {robot_batch.keys()}")
    print(f"Input tokens shape: {robot_batch['input_tokens'].shape}")
    print(f"Token types: {robot_batch['token_types'][:10]}...")
    
    print("\n✓ Data Loader tests passed!")
