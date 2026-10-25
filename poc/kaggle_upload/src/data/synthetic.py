"""
Synthetic Data Generation for Synobotix VLA PoC

Generates synthetic robotic sequences for testing and calibration.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from torch.utils.data import Dataset
import random


class SyntheticDataset(Dataset):
    """
    Dataset of synthetic text-vision sequences.
    
    Generates random but structured sequences of text and vision tokens
    for testing the VLA model.
    
    Args:
        num_samples: Number of samples in the dataset
        seq_length: Length of each sequence
        vocab_size: Size of text vocabulary
        image_size: Size of vision patches (num_patches)
        vision_ratio: Ratio of vision tokens in sequences
        metadata_ratio: Ratio of metadata tokens
    """
    
    def __init__(self, num_samples: int = 10000, seq_length: int = 100,
                 vocab_size: int = 50000, image_size: int = 256,
                 vision_ratio: float = 0.3, metadata_ratio: float = 0.1,
                 transform: Optional[Any] = None):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.vision_ratio = vision_ratio
        self.metadata_ratio = metadata_ratio
        self.transform = transform
        
        # Calculate number of each token type
        self.num_vision = int(seq_length * vision_ratio)
        self.num_metadata = int(seq_length * metadata_ratio)
        self.num_text = seq_length - self.num_vision - self.num_metadata
        
        # Vision parameters
        self.patch_size = 16
        self.num_channels = 3
        self.num_patches = (image_size // self.patch_size) ** 2
        self.vision_dim = 512  # Embedding dimension for vision
        
        # Generate all samples
        self.samples = []
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate all samples."""
        for _ in range(self.num_samples):
            sample = self._generate_sample()
            self.samples.append(sample)
    
    def _generate_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a single sample."""
        # Text tokens
        text_tokens = torch.randint(0, self.vocab_size, (self.num_text,))
        
        # Vision patches (simulated)
        vision_patches = torch.randn(self.num_vision, self.num_patches, 
                                     self.num_channels, self.patch_size, self.patch_size)
        
        # Metadata (simulated sensor data)
        metadata = torch.randn(self.num_metadata, 32)  # 32-dim metadata
        
        # Token types
        token_types = ['text'] * self.num_text + \
                      ['vision'] * self.num_vision + \
                      ['metadata'] * self.num_metadata
        
        # Target tokens (shifted for next token prediction)
        target_tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        
        sample = {
            'text_tokens': text_tokens,
            'vision_patches': vision_patches,
            'metadata': metadata,
            'token_types': token_types,
            'target_tokens': target_tokens
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
    
    def get_collate_fn(self):
        """
        Get a collate function for DataLoader.
        
        Returns:
            Function to collate batches
        """
        def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Stack text tokens
            text_tokens = [sample['text_tokens'] for sample in batch]
            max_text_len = max(len(t) for t in text_tokens)
            text_tokens_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
            for i, tokens in enumerate(text_tokens):
                text_tokens_padded[i, :len(tokens)] = tokens
            
            # Stack vision patches
            vision_patches = [sample['vision_patches'] for sample in batch]
            # Flatten patches for easier handling
            # [num_vision, num_patches, C, P, P] -> [num_vision, num_patches * C * P * P]
            vision_flat = []
            for patches in vision_patches:
                flat = patches.view(patches.shape[0], -1)
                vision_flat.append(flat)
            
            max_vision_len = max(len(v) for v in vision_flat)
            vision_flat_padded = torch.zeros(len(batch), max_vision_len, 
                                            self.num_patches * self.num_channels * 
                                            self.patch_size * self.patch_size)
            for i, flat in enumerate(vision_flat):
                vision_flat_padded[i, :flat.shape[0], :flat.shape[1]] = flat
            
            # Stack metadata
            metadata = [sample['metadata'] for sample in batch]
            max_metadata_len = max(len(m) for m in metadata)
            metadata_padded = torch.zeros(len(batch), max_metadata_len, 32)
            for i, meta in enumerate(metadata):
                metadata_padded[i, :len(meta)] = meta
            
            # Stack targets
            target_tokens = [sample['target_tokens'] for sample in batch]
            max_target_len = max(len(t) for t in target_tokens)
            target_tokens_padded = torch.zeros(len(batch), max_target_len, dtype=torch.long)
            for i, tokens in enumerate(target_tokens):
                target_tokens_padded[i, :len(tokens)] = tokens
            
            # Token types
            all_token_types = []
            for sample in batch:
                all_token_types.extend(sample['token_types'])
            
            return {
                'text_tokens': text_tokens_padded,
                'vision_patches': vision_flat_padded,
                'metadata': metadata_padded,
                'target_tokens': target_tokens_padded,
                'token_types': all_token_types
            }
        
        return collate_fn


class RoboticSequenceDataset(Dataset):
    """
    Dataset of realistic robotic manipulation sequences.
    
    Simulates sequences of robot actions, observations, and instructions.
    
    Args:
        num_sequences: Number of sequences
        seq_length: Length of each sequence
        num_actions: Number of possible robot actions
        observation_dim: Dimension of observation vectors
    """
    
    def __init__(self, num_sequences: int = 1000, seq_length: int = 200,
                 num_actions: int = 50, observation_dim: int = 64,
                 task_complexity: str = 'medium'):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self.task_complexity = task_complexity
        
        # Task types
        self.task_types = ['pick_and_place', 'assembly', 'inspection', 'navigation']
        
        # Generate sequences
        self.sequences = []
        self._generate_sequences()
    
    def _generate_sequences(self):
        """Generate all sequences."""
        for _ in range(self.num_sequences):
            seq = self._generate_sequence()
            self.sequences.append(seq)
    
    def _generate_sequence(self) -> Dict[str, Any]:
        """Generate a single robotic sequence."""
        # Randomly select a task
        task = random.choice(self.task_types)
        
        # Generate sequence based on task
        if task == 'pick_and_place':
            return self._generate_pick_and_place()
        elif task == 'assembly':
            return self._generate_assembly()
        elif task == 'inspection':
            return self._generate_inspection()
        else:
            return self._generate_navigation()
    
    def _generate_pick_and_place(self) -> Dict[str, Any]:
        """Generate a pick-and-place sequence."""
        seq_length = self.seq_length
        
        # Create a sequence with:
        # 1. Instruction (text)
        # 2. Observation (vision)
        # 3. Action (text/metadata)
        # 4. Result (vision)
        
        tokens = []
        token_types = []
        
        # Instruction: "Pick up object A and place it in location B"
        instruction_length = random.randint(10, 30)
        for _ in range(instruction_length):
            tokens.append(random.randint(0, self.num_actions))
            token_types.append('text')
        
        # Initial observation (vision)
        obs_length = random.randint(5, 15)
        for _ in range(obs_length):
            tokens.append(torch.randn(self.observation_dim))
            token_types.append('vision')
        
        # Actions
        num_actions = random.randint(3, 10)
        for _ in range(num_actions):
            tokens.append(random.randint(0, self.num_actions))
            token_types.append('text')
        
        # Final observation
        final_obs_length = random.randint(5, 15)
        for _ in range(final_obs_length):
            tokens.append(torch.randn(self.observation_dim))
            token_types.append('vision')
        
        # Pad or truncate to seq_length
        while len(tokens) < seq_length:
            if random.random() > 0.5:
                tokens.append(random.randint(0, self.num_actions))
                token_types.append('text')
            else:
                tokens.append(torch.randn(self.observation_dim))
                token_types.append('vision')
        
        tokens = tokens[:seq_length]
        token_types = token_types[:seq_length]
        
        # Create targets (shifted)
        targets = tokens[1:] + [random.randint(0, self.num_actions)]
        
        return {
            'tokens': tokens,
            'token_types': token_types,
            'targets': targets,
            'task': 'pick_and_place'
        }
    
    def _generate_assembly(self) -> Dict[str, Any]:
        """Generate an assembly sequence."""
        seq_length = self.seq_length
        
        tokens = []
        token_types = []
        
        # Complex instruction
        instruction_length = random.randint(15, 40)
        for _ in range(instruction_length):
            tokens.append(random.randint(0, self.num_actions))
            token_types.append('text')
        
        # Multiple observation-action pairs
        remaining = seq_length - instruction_length
        num_pairs = random.randint(3, 8)
        
        for _ in range(num_pairs):
            if remaining <= 0:
                break
            
            # Observation
            obs_length = min(random.randint(3, 10), remaining)
            for _ in range(obs_length):
                tokens.append(torch.randn(self.observation_dim))
                token_types.append('vision')
                remaining -= 1
            
            # Action
            if remaining > 0:
                tokens.append(random.randint(0, self.num_actions))
                token_types.append('text')
                remaining -= 1
        
        # Pad
        while len(tokens) < seq_length:
            tokens.append(random.randint(0, self.num_actions))
            token_types.append('text')
        
        tokens = tokens[:seq_length]
        token_types = token_types[:seq_length]
        
        targets = tokens[1:] + [random.randint(0, self.num_actions)]
        
        return {
            'tokens': tokens,
            'token_types': token_types,
            'targets': targets,
            'task': 'assembly'
        }
    
    def _generate_inspection(self) -> Dict[str, Any]:
        """Generate an inspection sequence."""
        seq_length = self.seq_length
        
        tokens = []
        token_types = []
        
        # Instruction
        instruction_length = random.randint(5, 20)
        for _ in range(instruction_length):
            tokens.append(random.randint(0, self.num_actions))
            token_types.append('text')
        
        # Many observations with few actions
        remaining = seq_length - instruction_length
        num_observations = random.randint(10, 30)
        
        for i in range(num_observations):
            if remaining <= 0:
                break
            
            obs_length = min(random.randint(1, 3), remaining)
            for _ in range(obs_length):
                tokens.append(torch.randn(self.observation_dim))
                token_types.append('vision')
                remaining -= 1
            
            # Occasionally add an action
            if i % 5 == 0 and remaining > 0:
                tokens.append(random.randint(0, self.num_actions))
                token_types.append('text')
                remaining -= 1
        
        # Pad
        while len(tokens) < seq_length:
            tokens.append(torch.randn(self.observation_dim))
            token_types.append('vision')
        
        tokens = tokens[:seq_length]
        token_types = token_types[:seq_length]
        
        targets = tokens[1:] + [random.randint(0, self.num_actions)]
        
        return {
            'tokens': tokens,
            'token_types': token_types,
            'targets': targets,
            'task': 'inspection'
        }
    
    def _generate_navigation(self) -> Dict[str, Any]:
        """Generate a navigation sequence."""
        seq_length = self.seq_length
        
        tokens = []
        token_types = []
        
        # Short instruction
        instruction_length = random.randint(3, 10)
        for _ in range(instruction_length):
            tokens.append(random.randint(0, self.num_actions))
            token_types.append('text')
        
        # Alternating observations and actions
        remaining = seq_length - instruction_length
        
        while remaining > 0:
            # Observation
            obs_length = min(random.randint(1, 5), remaining)
            for _ in range(obs_length):
                tokens.append(torch.randn(self.observation_dim))
                token_types.append('vision')
                remaining -= 1
            
            if remaining > 0:
                # Action
                tokens.append(random.randint(0, self.num_actions))
                token_types.append('text')
                remaining -= 1
        
        targets = tokens[1:] + [random.randint(0, self.num_actions)]
        
        return {
            'tokens': tokens,
            'token_types': token_types,
            'targets': targets,
            'task': 'navigation'
        }
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.sequences[idx]


class CalibrationDataset(Dataset):
    """
    Dataset for calibration of the quantizer.
    
    Provides diverse samples for computing statistics.
    """
    
    def __init__(self, num_samples: int = 10000, dim: int = 512):
        self.num_samples = num_samples
        self.dim = dim
        
        # Generate samples
        self.samples = []
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate calibration samples."""
        # Generate samples with different distributions
        for _ in range(self.num_samples):
            # Random distribution type
            dist_type = random.choice(['normal', 'uniform', 'sparse', 'bimodal'])
            
            if dist_type == 'normal':
                sample = torch.randn(self.dim)
            elif dist_type == 'uniform':
                sample = torch.rand(self.dim) * 2 - 1
            elif dist_type == 'sparse':
                sample = torch.zeros(self.dim)
                num_nonzero = random.randint(1, self.dim // 10)
                indices = random.sample(range(self.dim), num_nonzero)
                sample[indices] = torch.randn(num_nonzero) * 10
            else:  # bimodal
                sample = torch.randn(self.dim)
                # Add a secondary mode
                sample[:self.dim//2] += 5.0
            
            self.samples.append(sample)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


if __name__ == "__main__":
    print("Testing Synthetic Datasets...")
    
    # Test SyntheticDataset
    print("\n1. Testing SyntheticDataset:")
    dataset = SyntheticDataset(num_samples=10, seq_length=50, vocab_size=1000)
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Text tokens shape: {sample['text_tokens'].shape}")
    print(f"Vision patches shape: {sample['vision_patches'].shape}")
    print(f"Metadata shape: {sample['metadata'].shape}")
    print(f"Token types: {sample['token_types'][:10]}...")
    
    # Test collate function
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=dataset.get_collate_fn()
    )
    batch = next(iter(dataloader))
    print(f"Batch text tokens shape: {batch['text_tokens'].shape}")
    print(f"Batch vision patches shape: {batch['vision_patches'].shape}")
    
    # Test RoboticSequenceDataset
    print("\n2. Testing RoboticSequenceDataset:")
    robot_dataset = RoboticSequenceDataset(num_sequences=5, seq_length=30)
    robot_sample = robot_dataset[0]
    print(f"Robot sample keys: {robot_sample.keys()}")
    print(f"Task: {robot_sample['task']}")
    print(f"Token types distribution:")
    from collections import Counter
    type_counts = Counter(robot_sample['token_types'])
    for t, count in type_counts.items():
        print(f"  {t}: {count}")
    
    # Test CalibrationDataset
    print("\n3. Testing CalibrationDataset:")
    calib_dataset = CalibrationDataset(num_samples=100, dim=512)
    calib_sample = calib_dataset[0]
    print(f"Calibration sample shape: {calib_sample.shape}")
    print(f"Sample stats - mean: {calib_sample.mean():.4f}, std: {calib_sample.std():.4f}")
    
    print("\n✓ Synthetic Dataset tests passed!")
