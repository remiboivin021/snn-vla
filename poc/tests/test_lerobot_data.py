"""Tests for the LeRobot token dataset, using a fake dataset with the LeRobotDataset interface."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lerobot_data import FrameCache, LeRobotTokenDataset, TASK_VOCAB


class FakeLeRobotDataset:
    """Two episodes of 5 frames, state dim 2, action dim 2."""

    def __init__(self):
        self.decodes = 0
        self.rows = []
        for episode in range(2):
            for frame in range(5):
                self.rows.append({
                    'index': torch.tensor(len(self.rows)),
                    'episode_index': torch.tensor(episode),
                    'task_index': torch.tensor(episode),
                    'observation.state': torch.tensor([float(frame), 0.0]),
                    'action': torch.tensor([0.0, float(frame)]),
                })
        self.meta = SimpleNamespace(
            tasks=pd.DataFrame({'task_index': [0, 1]}, index=pd.Index(['push', 'pull'], name='task')),
            stats={'observation.state': {'min': [0.0, 0.0], 'max': [4.0, 1.0]},
                   'action': {'min': [0.0, 0.0], 'max': [1.0, 4.0]}},
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # Stands for the slow path: video decoding
        self.decodes += 1
        return {'observation.image': torch.full((3, 4, 4), idx / 10.0)}

    def get_raw_item(self, idx):
        return self.rows[idx]


class TestLeRobotTokenDataset(unittest.TestCase):

    def setUp(self):
        self.ds = LeRobotTokenDataset(FakeLeRobotDataset(), seq_length=40, vocab_size=1000)

    def test_shapes_and_shift(self):
        sample = self.ds[0]
        self.assertEqual(sample['text_tokens'].shape, (40,))
        self.assertEqual(sample['target_tokens'].shape, (40,))
        n = int((sample['target_tokens'] != -1).sum())
        self.assertTrue(torch.equal(sample['text_tokens'][1:n], sample['target_tokens'][:n - 1]))

    def test_task_bytes_then_binned_values(self):
        tokens = self.ds[0]['text_tokens'].tolist()
        self.assertEqual(tokens[:4], list(b'push'))
        # first frame: state [0, 0] -> bin 0 ; action [0, 0] -> bin 0
        self.assertEqual(tokens[4:8], [TASK_VOCAB] * 4)

    def test_window_stops_at_episode_end(self):
        # episode 0 = 4 task bytes + 5 frames * 4 tokens = 24 tokens -> 23 inputs
        sample = self.ds[0]
        self.assertEqual(int((sample['target_tokens'] != -1).sum()), 23)

    def test_tokens_stay_in_vocab(self):
        for i in range(len(self.ds)):
            sample = self.ds[i]
            self.assertLess(int(sample['text_tokens'].max()), 1000)
            self.assertLess(int(sample['target_tokens'].max()), 1000)

    def test_vocab_too_small(self):
        with self.assertRaises(ValueError):
            LeRobotTokenDataset(FakeLeRobotDataset(), vocab_size=TASK_VOCAB)


class TestFrameCache(unittest.TestCase):

    def setUp(self):
        self.fake = FakeLeRobotDataset()
        self.cache = FrameCache(tempfile.mkdtemp())
        self.ds = LeRobotTokenDataset(self.fake, seq_length=40, vocab_size=1000,
                                      image_keys=['observation.image'], frame_cache=self.cache)

    def test_frames_are_uint8(self):
        images = self.ds[4]['images']
        self.assertEqual(images.shape, (1, 3, 4, 4))
        self.assertEqual(images.dtype, torch.uint8)
        self.assertEqual(int(images[0, 0, 0, 0]), 102)  # 0.4 * 255

    def test_second_epoch_does_not_decode(self):
        first = [self.ds[i]['images'] for i in range(len(self.ds))]
        self.assertEqual(self.fake.decodes, len(self.ds))
        second = [self.ds[i]['images'] for i in range(len(self.ds))]
        self.assertEqual(self.fake.decodes, len(self.ds))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))

    def test_cache_survives_a_new_dataset_object(self):
        self.ds[0]
        again = LeRobotTokenDataset(FakeLeRobotDataset(), seq_length=40, vocab_size=1000,
                                    image_keys=['observation.image'], frame_cache=self.cache)
        again[0]
        self.assertEqual(again.dataset.decodes, 0)

    def test_no_images_unless_requested(self):
        plain = LeRobotTokenDataset(self.fake, seq_length=40, vocab_size=1000)
        self.assertNotIn('images', plain[0])
        self.assertEqual(self.fake.decodes, 0)


if __name__ == '__main__':
    unittest.main()
