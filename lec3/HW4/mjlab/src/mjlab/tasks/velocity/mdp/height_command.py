from __future__ import annotations

"""Height command term for pelvis/root absolute height.

Homework TODOs in this file: 1, 2  (of 10 total)
Index: docs/HOMEWORK_TODO.md · grep: 【作业 TODO
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformBaseHeightCommand(CommandTerm):
  """Uniformly sampled absolute pelvis/root height command above terrain."""

  cfg: UniformBaseHeightCommandCfg

  def __init__(self, cfg: UniformBaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_height_mean"] = torch.zeros(self.num_envs, device=self.device)

    self._height_slider: viser.GuiSliderHandle | None = None
    self._height_enabled: viser.GuiCheckboxHandle | None = None
    self._height_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    # >>> HOMEWORK_TODO_2_START
    # ==============================================================================
    # 【作业 TODO 2/10】高度跟踪误差指标
    # 位置: height_command.py · UniformBaseHeightCommand._update_metrics
    # 提示: 记录 |指令高度 - 实际高度| 的累积均值，用于训练日志与调试。
    # 概念: metrics["error_height"]、root_link_pos_w[:, 2]
    # 索引: docs/HOMEWORK_TODO.md
    # ==============================================================================
    raise NotImplementedError("TODO 2: 实现 height_error 与 metrics 更新")
    # --- 实现提示 ---
    # - 从 self.robot.data 读取骨盆世界坐标 z（root_link_pos_w[:, 2]）
    # - 计算 |指令高度 - 实际高度|，累加进 metrics["error_height"]（除以 max_command_step）
    # - 同步更新 metrics["target_height_mean"]
    # <<< HOMEWORK_TODO_2_END

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    # >>> HOMEWORK_TODO_1_START
    # ==============================================================================
    # 【作业 TODO 1/10】高度指令均匀采样
    # 位置: height_command.py · UniformBaseHeightCommand._resample_command
    # 提示: 在 cfg.ranges.height 范围内均匀随机采样新高度指令。
    # 概念: torch.uniform_、UniformBaseHeightCommandCfg.ranges
    # 索引: docs/HOMEWORK_TODO.md
    # ==============================================================================
    raise NotImplementedError("TODO 1: 实现 height_command 均匀采样")
    # --- 实现提示 ---
    # - 使用已创建的 r 张量，对 height_command[env_ids, 0] 做均匀随机采样
    # - 采样上下界来自 self.cfg.ranges.height
    # <<< HOMEWORK_TODO_1_END

  def _update_command(self) -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create a height slider in the Viser viewer."""
    height_range = self.cfg.ranges.height

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      slider = server.gui.add_slider(
        "height",
        min=height_range[0],
        max=height_range[1],
        step=0.01,
        initial_value=0.5 * (height_range[0] + height_range[1]),
      )

    self._height_enabled = enabled
    self._height_slider = slider
    self._height_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._height_enabled is not None and self._height_enabled.value:
      assert self._height_get_env_idx is not None
      idx = self._height_get_env_idx()
      assert self._height_slider is not None
      self.height_command[idx, 0] = self._height_slider.value

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw target and actual pelvis height markers."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    sphere_radius = 0.04 * visualizer.meansize

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      target_height = cmds[batch, 0]
      actual_height = base_pos_w[2]
      xy = base_pos_w[:2]

      target_center = np.array([xy[0], xy[1], target_height], dtype=np.float64)
      actual_center = np.array([xy[0] + 0.15, xy[1], actual_height], dtype=np.float64)

      visualizer.add_sphere(
        center=target_center,
        radius=sphere_radius,
        color=self.cfg.viz.target_color,
        label="base_height_target",
      )
      visualizer.add_sphere(
        center=actual_center,
        radius=sphere_radius,
        color=self.cfg.viz.actual_color,
        label="base_height_actual",
      )


@dataclass(kw_only=True)
class UniformBaseHeightCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.6, 0.0, 0.8)
    actual_color: tuple[float, float, float, float] = (0.2, 0.8, 1.0, 0.8)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformBaseHeightCommand:
    return UniformBaseHeightCommand(self, env)
