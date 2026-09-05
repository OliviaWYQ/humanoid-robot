"""Velocity task MDP observations.

Homework TODOs in this file: 10  (of 10 total)
Index: docs/HOMEWORK_TODO.md · grep: 【作业 TODO
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  # >>> HOMEWORK_TODO_10_START
  # ==============================================================================
  # 【作业 TODO 10/10】足部接触特权观测（Critic）
  # 位置: mdp/observations.py · foot_contact
  # 提示: 将 found>0 转为 0/1 浮点张量，供 critic 估计 value（actor 不可见）。
  # 概念: ContactSensor.data.found、privileged observations
  # 索引: docs/HOMEWORK_TODO.md
  # ==============================================================================
  raise NotImplementedError("TODO 10: 返回足部接触布尔/浮点张量")
  # --- 实现提示 ---
  # - 基于 sensor_data.found 构造 0/1 浮点张量（非 bool）
  # - shape 应为 [num_envs, num_feet]；供 critic 特权观测使用
  # <<< HOMEWORK_TODO_10_END


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
