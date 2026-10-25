"""A saved checkpoint must reload into the same architecture, whatever the config or last batch size."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vla_model import VLAModel


class TestCheckpointRoundTrip(unittest.TestCase):

    def test_non_default_config_reloads_with_same_outputs(self):
        root = Path(__file__).parent.parent
        config = yaml.safe_load(open(root / 'config' / 'model.yaml'))
        config['model'].update(hidden_dim=64, num_layers=3, num_heads=4, num_kv_heads=2)
        config['recurrent']['delta_net_hidden'] = 64
        tmp = Path(tempfile.mkdtemp())
        yaml.safe_dump(config, open(tmp / 'model.yaml', 'w'))

        model = VLAModel(config_path=str(tmp / 'model.yaml')).eval()
        tokens = torch.randint(0, 1000, (5, 12))
        model.delta_net.reset_hidden(5)  # leaves a batch-sized recurrent state, as after training
        expected = model(text_tokens=tokens)
        model.save(str(tmp / 'm.pt'))

        loaded = VLAModel.load(str(tmp / 'm.pt')).eval()
        loaded.delta_net.reset_hidden(5)
        self.assertEqual(loaded.dim, 64)
        self.assertEqual(len(loaded.layers), 3)
        model.delta_net.reset_hidden(5)
        self.assertTrue(torch.allclose(loaded(text_tokens=tokens), model(text_tokens=tokens), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
