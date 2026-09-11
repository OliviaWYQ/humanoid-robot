"""Run a small headless HW4 rollout; validates wiring, not policy performance.

From mjlab/: PYTHONPATH=src .venv/bin/python scripts/check_velocity_height.py
"""

import argparse
import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=2)
  args = parser.parse_args()
  if args.num_envs < 1 or args.steps < 1:
    parser.error("--num-envs and --steps must be positive")

  cfg = load_env_cfg("Mjlab-VelocityHeight-Flat-Unitree-G1")
  cfg.scene.num_envs = args.num_envs
  started = time.perf_counter()
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    obs, _ = env.reset(seed=42)
    for value in obs.values():
      assert torch.isfinite(value).all(), "Non-finite reset observation"
    command = env.command_manager.get_command("base_height")
    assert command is not None
    assert command.shape == (args.num_envs, 1)
    assert ((command >= 0.45) & (command <= 0.80)).all()
    action = torch.zeros(env.action_space.shape, device=args.device)
    for _ in range(args.steps):
      obs, reward, _, _, _ = env.step(action)
      assert torch.isfinite(reward).all(), "Non-finite reward"
      for value in obs.values():
        assert torch.isfinite(value).all(), "Non-finite step observation"
    height_reward = cfg.rewards["track_base_height"]
    tracking = height_reward.func(env, **height_reward.params)
    assert tracking.shape == (args.num_envs,)
    assert ((tracking >= 0) & (tracking <= 1)).all()
    print(
      {
        "device": args.device,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "observation_shapes": {k: tuple(v.shape) for k, v in obs.items()},
        "height_command": command.detach().cpu().tolist(),
        "height_reward": tracking.detach().cpu().tolist(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
      }
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
