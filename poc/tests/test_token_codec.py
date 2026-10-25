"""Training windows and closed-loop context must be built from the same token layout."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lerobot_data import ENV_STATE_KEY, STATE_KEY, LeRobotTokenDataset, TokenCodec

STATS = {
    ENV_STATE_KEY: {'min': [0.0] * 4, 'max': [10.0] * 4},
    STATE_KEY: {'min': [0.0, 0.0], 'max': [10.0, 10.0]},
    'action': {'min': [0.0, 0.0], 'max': [10.0, 10.0]},
}


class FakeDataset:
    """One episode of 6 steps with state-based observations."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.rows = [{'index': torch.tensor(i), 'episode_index': torch.tensor(0), 'task_index': torch.tensor(0),
                      ENV_STATE_KEY: torch.rand(4, generator=g) * 10, STATE_KEY: torch.rand(2, generator=g) * 10,
                      'action': torch.rand(2, generator=g) * 10} for i in range(6)]
        self.meta = SimpleNamespace(stats=STATS, tasks=pd.DataFrame(
            {'task_index': [0]}, index=pd.Index(['push'], name='task')))

    def __len__(self):
        return len(self.rows)

    def get_raw_item(self, i):
        return self.rows[i]


class TestTokenCodec(unittest.TestCase):

    def setUp(self):
        self.codec = TokenCodec.from_stats(STATS, vocab_size=1000)

    def test_uses_every_state_key_in_a_fixed_order(self):
        self.assertEqual(self.codec.state_keys, [ENV_STATE_KEY, STATE_KEY])
        self.assertEqual((self.codec.obs_len, self.codec.action_len, self.codec.step_len), (6, 2, 8))

    def test_action_round_trip_within_half_a_bin(self):
        action = torch.tensor([3.3, 7.7])
        decoded = self.codec.decode_action(self.codec.action_tokens(action))
        self.assertTrue(torch.all((decoded - action).abs() <= 10.0 / 255 / 2 + 1e-5))

    def test_training_window_equals_rollout_context(self):
        dataset = FakeDataset()
        window = LeRobotTokenDataset(dataset, seq_length=200, vocab_size=1000)[0]
        n = int((window['target_tokens'] != -1).sum())
        trained = window['text_tokens'][:n].tolist()

        # What a rollout feeds the model after 5 steps, observing step 5 (the last row)
        context = list(b'push')
        for row in dataset.rows[:5]:
            context += self.codec.obs_tokens([row[k] for k in self.codec.state_keys])
            context += self.codec.action_tokens(row['action'])
        last = dataset.rows[5]
        context += self.codec.obs_tokens([last[k] for k in self.codec.state_keys])
        self.assertEqual(trained[:len(context)], context)


if __name__ == '__main__':
    unittest.main()
