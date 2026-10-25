"""
Closed-loop PushT rollouts: does the trained model actually solve the task?

At every environment step the model reads the task, its recent history and the current
observation as tokens, then generates the two action tokens greedily. The action is
executed in gym-pusht and success is the environment's own (block coverage > 95%).
An "idle" policy (the agent never moves) gives the floor to compare against.

Needs a model trained on token sequences from a state-based PushT dataset:
    python scripts/train.py --dataset_type lerobot --repo_id lerobot/pusht_keypoints ...

Usage:
    python scripts/rollout_pusht.py --model_path outputs/train_X/model_best.pt \
        --episodes 50 --use_cuda
"""

import argparse
import json
import math
import os
import sys

# gym-pusht imports pygame: no display needed
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.lerobot_data import MAX_TASK_TOKENS, TASK_VOCAB, TokenCodec
from src.models.vla_model import VLAModel

ENV_ID = 'gym_pusht/PushT-v0'
WORLD_SIZE = 512.0


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% confidence interval of a success rate."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def observation_values(obs, state_keys):
    """Environment observation -> one array per state key, in the codec's order."""
    by_key = {'observation.environment_state': obs['environment_state'],
              'observation.state': obs['agent_pos']}
    return [by_key[key] for key in state_keys]


class ModelPolicy:
    """Generates the action tokens of one step from the token history."""

    def __init__(self, model, codec, task_tokens, history_steps, device):
        self.model, self.codec, self.device = model, codec, device
        self.prefix = task_tokens
        self.history_steps = history_steps
        self.history = []  # one token list (observation + action) per past step

    def reset(self):
        self.history = []

    @torch.no_grad()
    def act(self, obs):
        obs_tokens = self.codec.obs_tokens(observation_values(obs, self.codec.state_keys))
        context = list(self.prefix)
        for step_tokens in self.history[-self.history_steps:]:
            context += step_tokens
        context += obs_tokens

        action_tokens = []
        for _ in range(self.codec.action_len):
            tokens = torch.tensor([context + action_tokens], device=self.device)
            self.model.delta_net.reset_hidden(1)
            logits = self.model(text_tokens=tokens)[0, -1]
            # Only action bins are valid here
            bins = logits[TASK_VOCAB:TASK_VOCAB + self.codec.num_bins]
            action_tokens.append(TASK_VOCAB + int(bins.argmax()))

        self.history.append(obs_tokens + action_tokens)
        return self.codec.decode_action(action_tokens).clamp(0, WORLD_SIZE).numpy()


class IdlePolicy:
    """The agent stays where it is: the floor any learned policy must beat."""

    def reset(self):
        pass

    def act(self, obs):
        return obs['agent_pos'].copy()


def run_episode(env, policy, seed, max_steps):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    best_coverage, success = 0.0, False
    for _ in range(max_steps):
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        best_coverage = max(best_coverage, float(info['coverage']))
        success = bool(info['is_success'])
        if terminated or truncated:
            break
    return {'seed': seed, 'success': success, 'max_coverage': best_coverage}


def summarize(name, episodes):
    n = len(episodes)
    successes = sum(e['success'] for e in episodes)
    low, high = wilson_interval(successes, n)
    return {'policy': name, 'episodes': n, 'successes': successes,
            'success_rate': successes / n, 'success_ci95': [low, high],
            'mean_max_coverage': sum(e['max_coverage'] for e in episodes) / n}


def main():
    parser = argparse.ArgumentParser(description='Closed-loop PushT evaluation')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--repo_id', default='lerobot/pusht_keypoints')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--history_steps', type=int, default=47,
                        help='Past steps kept in context (task + history must fit the training length)')
    parser.add_argument('--output_dir', default='rollout')
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the environment)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    model = VLAModel.load(args.model_path).to(device).eval()
    meta = LeRobotDatasetMetadata(args.repo_id)
    codec = TokenCodec.from_stats(meta.stats, model.text_vocab_size)
    task_tokens = list(meta.tasks.iloc[0].name.encode('utf-8'))[:MAX_TASK_TOKENS]

    env = gym.make(ENV_ID, obs_type='environment_state_agent_pos', max_episode_steps=args.max_steps)
    policies = [('idle (agent never moves)', IdlePolicy()),
                ('model, fp16 KV cache', ModelPolicy(model, codec, task_tokens, args.history_steps, device))]

    summaries, detail = [], {}
    for name, policy in policies:
        print(f'\n== {name}', flush=True)
        episodes = []
        for seed in range(args.episodes):
            episodes.append(run_episode(env, policy, seed, args.max_steps))
            done = sum(e['success'] for e in episodes)
            print(f'  episode {seed + 1}/{args.episodes}: coverage {episodes[-1]["max_coverage"]:.2f} '
                  f'| successes so far {done}', flush=True)
        detail[name] = episodes
        summaries.append(summarize(name, episodes))

    print('\n' + '=' * 78)
    print(f'{"policy":<30}{"success":>10}{"95% CI":>18}{"mean best coverage":>20}')
    for r in summaries:
        ci = f'[{r["success_ci95"][0]:.2f}, {r["success_ci95"][1]:.2f}]'
        print(f'{r["policy"]:<30}{r["successes"]:>4}/{r["episodes"]:<5}{ci:>18}{r["mean_max_coverage"]:>20.3f}')
    print('=' * 78)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'rollout.json'), 'w') as f:
        json.dump({'args': vars(args), 'summary': summaries, 'episodes': detail}, f, indent=2)


if __name__ == '__main__':
    main()
