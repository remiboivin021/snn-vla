"""
LeRobot datasets for Synobotix VLA PoC.

Wraps a LeRobotDataset (https://github.com/huggingface/lerobot) as next-token
sequences for the VLA model. Only the text-token path is used, so images are never
decoded: rows are read with LeRobotDataset.get_raw_item().

Token layout (vocab_size must be >= TASK_VOCAB + num_bins):
    [0, 256)               task instruction, UTF-8 bytes
    [256, 256 + num_bins)  state / action values, uniformly binned per dimension
Sequence: task bytes, then for each consecutive frame of the episode its state
bins followed by its action bins. Targets are the sequence shifted by one;
padding positions get target -1 (the ignore_index of the training loss).
"""

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

TASK_VOCAB = 256
MAX_TASK_TOKENS = 64
STATE_KEY = 'observation.state'
ACTION_KEY = 'action'


def make_lerobot_dataset(repo_id: str, split: str = 'train', val_fraction: float = 0.1,
                         root: str = None):
    """
    Build a LeRobotDataset restricted to the train or val episodes.
    
    Splitting by episode keeps windows of the same episode out of both splits.
    """
    # Imported lazily: lerobot is only needed for this dataset type
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    total = LeRobotDatasetMetadata(repo_id, root=root).total_episodes
    num_val = max(1, int(total * val_fraction)) if total > 1 else 0
    episodes = list(range(total - num_val, total)) if split != 'train' else list(range(total - num_val))
    # Images are not used by the text-token path: skip the video download
    return LeRobotDataset(repo_id, root=root, episodes=episodes or None, download_videos=False)


class LeRobotTokenDataset(Dataset):
    """
    Next-token sequences built from consecutive frames of a LeRobot dataset.
    
    Args:
        dataset: LeRobotDataset (anything with get_raw_item, meta.tasks, meta.stats, __len__)
        seq_length: Number of input positions per sample
        vocab_size: Size of the model vocabulary
    """

    def __init__(self, dataset, seq_length: int = 100, vocab_size: int = 50000):
        num_bins = vocab_size - TASK_VOCAB
        if num_bins < 2:
            raise ValueError(f"vocab_size must be > {TASK_VOCAB + 1}, got {vocab_size}")
        self.dataset = dataset
        self.seq_length = seq_length
        self.num_bins = min(num_bins, 256)

        stats = dataset.meta.stats
        self.state_min, self.state_max = self._range(stats[STATE_KEY])
        self.action_min, self.action_max = self._range(stats[ACTION_KEY])

    @staticmethod
    def _range(stat: Dict[str, Any]):
        return torch.as_tensor(stat['min'], dtype=torch.float32), \
               torch.as_tensor(stat['max'], dtype=torch.float32)

    def _bin(self, values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> List[int]:
        values = torch.as_tensor(values, dtype=torch.float32).flatten()
        unit = ((values - low) / (high - low).clamp(min=1e-8)).clamp(0, 1)
        return (TASK_VOCAB + (unit * (self.num_bins - 1)).round().long()).tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        first = self.dataset.get_raw_item(idx)
        episode = int(first['episode_index'])
        task = self.dataset.meta.tasks.iloc[int(first['task_index'])].name
        tokens = list(task.encode('utf-8'))[:MAX_TASK_TOKENS]

        # +1 token: the last one is only a target
        row, i = first, idx
        while len(tokens) < self.seq_length + 1:
            tokens += self._bin(row[STATE_KEY], self.state_min, self.state_max)
            tokens += self._bin(row[ACTION_KEY], self.action_min, self.action_max)
            i += 1
            if i >= len(self.dataset):
                break
            row = self.dataset.get_raw_item(i)
            if int(row['episode_index']) != episode:
                break

        tokens = tokens[:self.seq_length + 1]
        inputs = torch.zeros(self.seq_length, dtype=torch.long)
        targets = torch.full((self.seq_length,), -1, dtype=torch.long)
        n = len(tokens) - 1
        inputs[:n] = torch.tensor(tokens[:-1])
        targets[:n] = torch.tensor(tokens[1:])
        return {'text_tokens': inputs, 'target_tokens': targets}
