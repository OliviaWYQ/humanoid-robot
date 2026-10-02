"""CPU regression tests: real HW7 data, AMP rewards, episode history and PPO updates.

Run from the project root: python -m unittest discover -s tests -v
No Isaac Sim installation is needed for these tests.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsl_rl_amp.algorithms.amp import AMP
from rsl_rl_amp.algorithms.discriminator import AMPDiscriminator
from rsl_rl_amp.env import AMPIsaacLabVecEnvWrapper
from rsl_rl_amp.models import MLPModel
from rsl_rl_amp.storage import RolloutStorage

# Load the portable data modules without executing the Isaac extension initializer.
AMP_ROOT = ROOT / 'source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp'
for name, path in [('hw8_data', AMP_ROOT), ('hw8_data.config', AMP_ROOT / 'config'),
                   ('hw8_data.config.g1', AMP_ROOT / 'config/g1')]:
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
from hw8_data.motion_dataset import DEFAULT_FEATURES, MotionDataset
from hw8_data.config.g1.motion_cfg import G1WalkToRunMotionCfg, G1_REFERENCE_JOINT_NAMES, make_motion_dataset_cfg


def dataset():
    cfg = make_motion_dataset_cfg(G1WalkToRunMotionCfg())
    cfg.joint_names = G1_REFERENCE_JOINT_NAMES
    return MotionDataset(cfg, 'cpu')


class RewardTests(unittest.TestCase):
    def test_lsq_boundaries_and_time_scale(self):
        disc = AMPDiscriminator(80, 3, hidden_dims=[16], reward_scale=5)
        scores = torch.tensor([-3., -1., 0., 1., 3., 5.])
        with patch.object(disc, 'forward', return_value=scores):
            reward, raw = disc.style_reward(torch.zeros(6, 3, 80), .02)
            torch.testing.assert_close(reward, torch.tensor([0., 0., .075, .1, 0., 0.]))
            torch.testing.assert_close(raw, scores)
            doubled, _ = disc.style_reward(torch.zeros(6, 3, 80), .04)
            torch.testing.assert_close(doubled, 2 * reward)
        with self.assertRaises(ValueError):
            disc.style_reward(torch.zeros(1, 240), 0.)

    def test_mixing_endpoints_and_shapes(self):
        style, task = torch.tensor([.1, .05]), torch.tensor([-1., 2.])
        for weight in [0., .6, 1.]:
            disc = AMPDiscriminator(80, 3, hidden_dims=[16], task_reward_weight=weight)
            for a, b in [(style, task), (style[:, None], task[:, None])]:
                result = disc.mix_rewards(a, b)
                self.assertEqual(result.shape, b.shape)
                torch.testing.assert_close(result, (1 - weight) * a + weight * b)
        with self.assertRaises(ValueError):
            disc.mix_rewards(style, task[:, None])


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(8)
        torch.set_num_threads(1)
        cls.expert = dataset()

    def test_real_data_and_joint_reordering(self):
        data = self.expert
        self.assertEqual(data.frame_dim, 80)
        self.assertEqual(tuple(data.feature_slices), DEFAULT_FEATURES)
        self.assertEqual(len(data.clips), 4)
        samples = data.sample(32)
        self.assertEqual(samples.shape, (32, 240))
        self.assertTrue(torch.isfinite(samples).all())
        cfg = make_motion_dataset_cfg(G1WalkToRunMotionCfg())
        cfg.joint_names = tuple(reversed(G1_REFERENCE_JOINT_NAMES))
        reordered = MotionDataset(cfg, 'cpu')
        for original, reverse in zip(data.clips, reordered.clips):
            torch.testing.assert_close(reverse.joint_pos, original.joint_pos.flip(-1))
            torch.testing.assert_close(reverse.joint_vel, original.joint_vel.flip(-1))
        state = data.sample_reference_state(16)
        self.assertEqual(state.joint_pos.shape, (16, 29))
        torch.testing.assert_close(state.root_quat.norm(dim=-1), torch.ones(16))

    def test_joint_ppo_discriminator_update_and_history_reset(self):
        n, steps = 4, 8
        def observation():
            return TensorDict({'policy': torch.randn(n, 8), 'critic': torch.randn(n, 10),
                               'amp': self.expert.sample(n).reshape(n, 3, 80)}, batch_size=[n])
        obs = observation()
        groups = {'actor': ['policy'], 'critic': ['critic'], 'amp': ['amp']}
        actor = MLPModel(obs, groups, 'actor', 2, hidden_dims=[16], distribution_cfg={
            'class_name': 'rsl_rl_amp.modules.GaussianDistribution', 'init_std': .8})
        critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[16])
        storage = RolloutStorage('rl', n, steps, obs, [2])
        alg = AMP(actor, critic, storage, num_learning_epochs=1, num_mini_batches=2,
                  schedule='fixed', amp_cfg={
            'observation_groups': ['amp'], 'frame_dim': 80, 'history_steps': 3,
            'step_dt': .02, 'expert_sampler': self.expert, 'motion_profile': 'walk_to_run',
            'replay_buffer_size': 128, 'discriminator_updates': 1,
            'discriminator_batch_size': 8, 'normalization_batch_size': 8,
            'discriminator_balance': {'enabled': False},
            'discriminator': {'hidden_dims': [16], 'reward_scale': 5., 'task_reward_weight': .6},
        })
        before_actor = [p.detach().clone() for p in actor.parameters()]
        before_disc = [p.detach().clone() for p in alg.discriminator.network.parameters()]
        ages = torch.zeros(n, dtype=torch.long)
        expected_count = 0
        for step in range(steps):
            with torch.no_grad():
                alg.act(obs)
                obs = observation()
                task = torch.tensor([1., -1., .2, .4])
                done = torch.tensor([step == 3, False, step == 5, False])
                valid = ages + 1 >= 3
                alg.process_env_step(obs, task, done, {})
            expected = torch.where(valid, .4 * alg.last_style_rewards + .6 * task, task)
            torch.testing.assert_close(storage.rewards[step, :, 0], expected)
            expected_count += int(valid.sum())
            self.assertEqual(len(alg.replay_buffer), expected_count)
            ages = torch.where(done, 0, ages + 1)
            torch.testing.assert_close(alg._amp_history_age, ages)
        alg.compute_returns(obs)
        losses = alg.update()
        self.assertTrue(all(math.isfinite(v) for v in losses.values()))
        self.assertEqual(losses['amp_discriminator_updates'], 1.)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_actor, actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_disc, alg.discriminator.network.parameters())))
        for model in [actor, critic, alg.discriminator]:
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class HistoryTests(unittest.TestCase):
    def test_terminal_frame_and_next_episode_are_separate(self):
        class Env:
            num_envs = 2
            device = 'cpu'
            max_episode_length = 100
            cfg = SimpleNamespace(amp_history_steps=3, is_finite_horizon=False)
            action_manager = SimpleNamespace(total_action_dim=1)
            terminal_amp_frames = torch.tensor([[9.], [0.]])
            @property
            def unwrapped(self):
                return self
            def reset(self):
                return {'amp': torch.tensor([[1.], [2.]])}, {}
            def step(self, actions):
                return ({'amp': torch.tensor([[100.], [3.]])}, torch.ones(2),
                        torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool), {})
        wrapper = AMPIsaacLabVecEnvWrapper(Env())
        transition, _, _, _ = wrapper.step(torch.zeros(2, 1))
        torch.testing.assert_close(transition['amp'][0, :, 0], torch.tensor([1., 1., 9.]))
        torch.testing.assert_close(wrapper.get_observations()['amp'][0, :, 0], torch.tensor([100., 100., 100.]))
        torch.testing.assert_close(transition['amp'][1, :, 0], torch.tensor([2., 2., 3.]))


if __name__ == '__main__':
    unittest.main()
