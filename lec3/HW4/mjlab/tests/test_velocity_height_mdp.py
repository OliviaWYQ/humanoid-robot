"""CPU checks for HW4 command sampling, tracking rewards and curriculum."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity import mdp

TASK_ID = "Mjlab-VelocityHeight-Flat-Unitree-G1"


def test_height_sampling_metrics_and_reset_isolation():
  cfg = load_env_cfg(TASK_ID).commands["base_height"]
  env = SimpleNamespace(
    num_envs=4096,
    device="cpu",
    step_dt=0.02,
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          root_link_pos_w=torch.zeros(4096, 3),
        )
      )
    },
  )
  term = cfg.build(env)
  ids = torch.arange(0, env.num_envs, 2)
  with torch.random.fork_rng():
    torch.manual_seed(42)
    term._resample_command(ids)
  sampled = term.command[ids, 0]
  assert sampled.min() >= 0.45
  assert sampled.max() <= 0.80
  assert abs(sampled.mean().item() - 0.625) < 0.01
  assert torch.count_nonzero(term.command[1::2]) == 0
  previous = term.command.clone()
  term._resample_command(torch.empty(0, dtype=torch.long))
  torch.testing.assert_close(term.command, previous)

  term.height_command.fill_(0.5)
  env.scene["robot"].data.root_link_pos_w[:, 2] = 0.6
  term._update_metrics()
  term._update_metrics()
  torch.testing.assert_close(
    term.metrics["error_height"],
    torch.full((4096,), 0.0005),
  )
  torch.testing.assert_close(
    term.metrics["target_height_mean"], torch.full((4096,), 0.5)
  )
  # CommandManager clears metric buffers at reset. They must not alias commands.
  term.metrics["target_height_mean"].zero_()
  torch.testing.assert_close(term.command, torch.full((4096, 1), 0.5))


def test_height_reward_uses_absolute_world_z_and_squared_error():
  env = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          root_link_pos_w=torch.tensor(
            [[10.0, -7.0, 0.5], [0.0, 0.0, 0.4], [0.0, 0.0, 0.7]]
          ),
        )
      )
    },
    command_manager=SimpleNamespace(get_command=lambda _: torch.full((3, 1), 0.5)),
  )
  result = mdp.track_base_height(env, std=0.1, command_name="base_height")
  torch.testing.assert_close(result, torch.tensor([1.0, math.exp(-1), math.exp(-4)]))


@pytest.mark.parametrize(
  "reward,field,actual",
  [
    (
      mdp.track_linear_velocity,
      "root_link_lin_vel_b",
      [[0.3, -0.2, 0.0], [0.3, -0.2, 0.5], [0.6, 0.2, 0.0]],
    ),
    (
      mdp.track_angular_velocity,
      "root_link_ang_vel_b",
      [[0.0, 0.0, 0.7], [0.3, 0.4, 0.7], [0.0, 0.0, 0.2]],
    ),
  ],
)
def test_velocity_rewards_penalize_uncommanded_axes(reward, field, actual):
  env = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(data=SimpleNamespace(**{field: torch.tensor(actual)}))
    },
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor([[0.3, -0.2, 0.7]]).repeat(3, 1),
    ),
  )
  result = reward(env, std=0.5, command_name="velocity")
  torch.testing.assert_close(result, torch.tensor([1.0, math.exp(-1), math.exp(-1)]))


def test_contact_is_binary_float_and_critic_only():
  env = SimpleNamespace(
    scene={
      "feet": SimpleNamespace(
        data=SimpleNamespace(
          found=torch.tensor([[0, 1], [2, 0], [3, 4]]),
        )
      )
    }
  )
  result = mdp.foot_contact(env, "feet")
  torch.testing.assert_close(result, torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
  cfg = load_env_cfg(TASK_ID)
  assert "foot_contact" not in cfg.observations["actor"].terms
  assert "foot_contact" in cfg.observations["critic"].terms


@pytest.mark.parametrize("play", [False, True])
def test_height_command_is_wired_to_both_observation_groups_and_reward(play):
  cfg = load_env_cfg(TASK_ID, play=play)
  for group in ("actor", "critic"):
    term = cfg.observations[group].terms["height_command"]
    assert term.func is mdp.generated_commands
    assert term.params["command_name"] == "base_height"
  reward = cfg.rewards["track_base_height"]
  assert reward.func is mdp.track_base_height
  assert reward.params["command_name"] == "base_height"
  assert reward.weight == 1.0


@pytest.mark.parametrize(
  "step,x_range,yaw_range",
  [
    (0, (-1.0, 1.0), (-0.5, 0.5)),
    (119999, (-1.0, 1.0), (-0.5, 0.5)),
    (120000, (-1.5, 2.0), (-0.7, 0.7)),
    (239999, (-1.5, 2.0), (-0.7, 0.7)),
    (240000, (-2.0, 3.0), (-0.7, 0.7)),
  ],
)
def test_velocity_curriculum_boundaries_and_inherited_yaw(step, x_range, yaw_range):
  cfg = load_env_cfg(TASK_ID)
  velocity = cfg.commands["velocity"]
  env = SimpleNamespace(
    common_step_counter=step,
    command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(cfg=velocity)),
  )
  curriculum = cfg.curriculum["command_vel"]
  curriculum.func(env, torch.tensor([0]), **curriculum.params)
  assert velocity.ranges.lin_vel_x == x_range
  assert velocity.ranges.ang_vel_z == yaw_range
  assert velocity.ranges.lin_vel_y == (-1.0, 1.0)
