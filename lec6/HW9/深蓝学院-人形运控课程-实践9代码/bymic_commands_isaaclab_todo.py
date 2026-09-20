from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device
        )
        self.joint_vel = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device
        )
        self._body_pos_w = torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device
        )
        self._body_quat_w = torch.tensor(
            data["body_quat_w"], dtype=torch.float32, device=device
        )
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = MotionLoader(
            self.cfg.motion_file, self.body_indexes, device=self.device
        )
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = (
            int(
                self.motion.time_step_total
                // (1 / (env.cfg.decimation * env.cfg.sim.dt))
            )
            + 1
        )
        self.bin_failed_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_rot"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_prob"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["sampling_top1_bin"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps]
            + self._env.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[
            self.time_steps, self.motion_anchor_body_index
        ]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        ### TODO: Implement Adaptive Motion Sampling / 实现自适应动作采样
        # 1. Read failed environments / 读取失败环境:
        #    Use `self._env.termination_manager.terminated[env_ids]` to get `episode_failed`.
        #    `env_ids` is the list/tensor of reset environments; `episode_failed` must be a bool mask.
        #    使用 `self._env.termination_manager.terminated[env_ids]` 得到 `episode_failed`。
        #    `env_ids` 是要重置的环境编号；`episode_failed` 必须是布尔掩码。
        # 2. Map time steps to bins / 把动作帧映射到采样区间:
        #    Convert `self.time_steps` into `current_bin_index` with `self.bin_count`.
        #    Formula hint: `(self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1)`.
        #    Clamp the result to `[0, self.bin_count - 1]`.
        #    用 `self.bin_count` 把 `self.time_steps` 转成 `current_bin_index`。
        #    公式提示：`(self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1)`。
        #    最后用 `torch.clamp` 限制到 `[0, self.bin_count - 1]`。
        # 3. Count failures per bin / 统计每个区间失败次数:
        #    Select failed bins with `current_bin_index[env_ids][episode_failed]`.
        #    Use `torch.bincount(..., minlength=self.bin_count)` and write into `self._current_bin_failed[:]`.
        #    用 `current_bin_index[env_ids][episode_failed]` 取出失败环境所在bin。
        #    用 `torch.bincount(..., minlength=self.bin_count)` 统计，并写入 `self._current_bin_failed[:]`。
        # 4. Build base probabilities / 构造基础采样概率:
        #    Start from `self.bin_failed_count`.
        #    Add `self.cfg.adaptive_uniform_ratio / float(self.bin_count)` so every bin remains sampleable.
        #    从 `self.bin_failed_count` 开始。
        #    加上 `self.cfg.adaptive_uniform_ratio / float(self.bin_count)`，避免某些bin概率为0。
        # 5. Smooth probabilities / 平滑概率:
        #    Reshape to `(1, 1, self.bin_count)`, pad right by `self.cfg.adaptive_kernel_size - 1`,
        #    then apply `torch.nn.functional.conv1d(..., self.kernel.view(1, 1, -1))`.
        #    Key parameters: `adaptive_kernel_size`, `adaptive_lambda`, `self.kernel`.
        #    先 reshape 成 `(1, 1, self.bin_count)`，右侧 padding 长度为 `self.cfg.adaptive_kernel_size - 1`，
        #    再用 `torch.nn.functional.conv1d(..., self.kernel.view(1, 1, -1))` 做平滑。
        #    关键参数：`adaptive_kernel_size`、`adaptive_lambda`、`self.kernel`。
        # 6. Normalize probabilities / 归一化概率:
        #    Divide by `sampling_probabilities.sum()`; the final probabilities must sum to 1.
        #    除以 `sampling_probabilities.sum()`；最终所有概率之和必须为1。
        # 7. Sample bins / 采样动作区间:
        #    Use `torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)`.
        #    `sampled_bins` must have shape `(len(env_ids),)`.
        #    用 `torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)`。
        #    `sampled_bins` 形状必须是 `(len(env_ids),)`。
        # 8. Convert bins to time steps / 把bin转回动作帧:
        #    Add random offsets from `sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)`.
        #    Convert to `[0, self.motion.time_step_total - 1]`, cast to `.long()`, and write `self.time_steps[env_ids]`.
        #    加入 `sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)` 产生的随机偏移。
        #    转成 `[0, self.motion.time_step_total - 1]` 内的帧编号，`.long()` 后写入 `self.time_steps[env_ids]`。
        # 9. Update metrics / 更新采样指标:
        #    Compute entropy `H`, normalized entropy `H_norm`, max probability `pmax`, and max-probability bin `imax`.
        #    Store them in `sampling_entropy`, `sampling_top1_prob`, and `sampling_top1_bin`.
        #    计算熵 `H`、归一化熵 `H_norm`、最大概率 `pmax`、最大概率bin `imax`。
        #    写入 `sampling_entropy`、`sampling_top1_prob`、`sampling_top1_bin`。
        raise NotImplementedError(
            "TODO: implement _adaptive_sampling / 请完成 _adaptive_sampling"
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            *self.cfg.joint_position_range, joint_pos.shape, joint_pos.device
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_ori[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

    def _update_command(self):
        ### TODO: Implement Command Update / 实现命令逐步更新
        # 1. Advance motion time / 推进动作时间:
        #    Increase every environment time step with `self.time_steps += 1`.
        #    用 `self.time_steps += 1` 让每个环境的动作帧前进一帧。
        # 2. Resample finished motions / 重采样播放结束的动作:
        #    Find environments where `self.time_steps >= self.motion.time_step_total`.
        #    Use `torch.where(...)[0]` to get `env_ids`, then call `self._resample_command(env_ids)`.
        #    找出 `self.time_steps >= self.motion.time_step_total` 的环境。
        #    用 `torch.where(...)[0]` 得到 `env_ids`，然后调用 `self._resample_command(env_ids)`。
        # 3. Repeat anchor tensors / 扩展anchor张量:
        #    `body_count = len(self.cfg.body_names)`.
        #    Repeat `self.anchor_pos_w`, `self.anchor_quat_w`,
        #    `self.robot_anchor_pos_w`, and `self.robot_anchor_quat_w`.
        #    Position repeat shape: `(self.num_envs, body_count, 3)`.
        #    Quaternion repeat shape: `(self.num_envs, body_count, 4)`.
        #    `body_count = len(self.cfg.body_names)`。
        #    扩展 `self.anchor_pos_w`、`self.anchor_quat_w`、
        #    `self.robot_anchor_pos_w`、`self.robot_anchor_quat_w`。
        #    位置张量形状：`(self.num_envs, body_count, 3)`。
        #    四元数张量形状：`(self.num_envs, body_count, 4)`。
        # 4. Compute position alignment / 计算位置对齐:
        #    Start `delta_pos_w` from `robot_anchor_pos_w_repeat`.
        #    Replace only the z value with `anchor_pos_w_repeat[..., 2]`.
        #    This keeps robot x/y but uses the reference motion height.
        #    从 `robot_anchor_pos_w_repeat` 得到 `delta_pos_w`。
        #    只把 z 值替换成 `anchor_pos_w_repeat[..., 2]`。
        #    这样 x/y 跟随机器人当前位置，z 使用参考动作高度。
        # 5. Compute yaw-only rotation alignment / 计算只包含yaw的旋转对齐:
        #    Use `quat_inv(anchor_quat_w_repeat)` to invert the reference anchor orientation.
        #    Use `quat_mul(robot_anchor_quat_w_repeat, quat_inv(...))` to get the rotation difference.
        #    Wrap it with `yaw_quat(...)` so only heading/yaw is kept.
        #    用 `quat_inv(anchor_quat_w_repeat)` 求参考anchor姿态的逆。
        #    用 `quat_mul(robot_anchor_quat_w_repeat, quat_inv(...))` 得到旋转差。
        #    再用 `yaw_quat(...)` 只保留朝向/yaw。
        # 6. Build aligned target body poses / 构造对齐后的目标身体位姿:
        #    `self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)`.
        #    `self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)`.
        #    Key outputs: `self.body_quat_relative_w` and `self.body_pos_relative_w`.
        #    `self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)`。
        #    `self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)`。
        #    关键输出：`self.body_quat_relative_w` 和 `self.body_pos_relative_w`。
        # 7. Update adaptive failure history / 更新自适应失败历史:
        #    Use exponential moving average:
        #    `bin_failed_count = adaptive_alpha * current_failures + (1 - adaptive_alpha) * old_failures`.
        #    Key parameter: `self.cfg.adaptive_alpha`.
        #    使用指数滑动平均：
        #    `bin_failed_count = adaptive_alpha * current_failures + (1 - adaptive_alpha) * old_failures`。
        #    关键参数：`self.cfg.adaptive_alpha`。
        # 8. Clear temporary failure counts / 清空临时失败计数:
        #    Call `self._current_bin_failed.zero_()` after updating `self.bin_failed_count`.
        #    更新 `self.bin_failed_count` 后，调用 `self._current_bin_failed.zero_()`。
        raise NotImplementedError(
            "TODO: implement _update_command / 请完成 _update_command"
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/current/anchor"
                    )
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(
                        prim_path="/Visuals/Command/goal/anchor"
                    )
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/current/" + name
                            )
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(
                                prim_path="/Visuals/Command/goal/" + name
                            )
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
            else:
                return
            
    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(
            self.robot_anchor_pos_w, self.robot_anchor_quat_w
        )
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i]
            )
            self.goal_body_visualizers[i].visualize(
                self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i]
            )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
