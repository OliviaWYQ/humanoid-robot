"""Check HW8 configuration and AMP observations in a real Isaac Lab environment."""
import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'source/unitree_rl_lab'))

try:
    import gymnasium as gym
    import torch
    import unitree_rl_lab.tasks  # noqa: F401
    from rsl_rl_amp.env import AMPIsaacLabVecEnvWrapper
    from rsl_rl_amp.runners.amp_runner import validate_amp_motion_profile
    from unitree_rl_lab.tasks.locomotion.amp.motion_dataset import DEFAULT_FEATURES
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task = 'Unitree-G1-29dof-AMP-WalkToRun-FullPlay'
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs,
                        entry_point_key='play_env_cfg_entry_point')
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    assert cfg.curriculum.lin_vel_cmd_levels is None
    assert cfg.curriculum.ang_vel_cmd_levels is None
    assert not cfg.observations.policy.enable_corruption
    assert cfg.events.push_robot is None and cfg.events.base_external_force_torque is None
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-1.0, 4.2)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (-1.2, 1.2)
    env = gym.make(task, cfg=cfg)
    try:
        wrapper = AMPIsaacLabVecEnvWrapper(env)
        assert validate_amp_motion_profile(wrapper, agent.to_dict()) == 'walk_to_run'
        assert tuple(env.unwrapped.observation_manager.active_terms['amp']) == DEFAULT_FEATURES
        obs = wrapper.get_observations()
        assert obs['amp'].shape == (args.num_envs, 3, 80)
        expert = wrapper.amp_expert_sampler.sample(args.num_envs)
        assert expert.shape == (args.num_envs, 240) and torch.isfinite(expert).all()
        for _ in range(4):
            obs, rewards, _, _ = wrapper.step(torch.zeros(
                args.num_envs, wrapper.num_actions, device=wrapper.device))
            assert torch.isfinite(obs['amp']).all() and torch.isfinite(rewards).all()
        print('HW8 Isaac Lab check passed: profile, FullPlay, 80/240 dimensions, data and environment steps.')
    finally:
        env.close()
finally:
    launcher.app.close()
