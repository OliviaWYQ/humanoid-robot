"""Explicit and student-editable G1 AMP train/Play environment configurations."""

from __future__ import annotations

import importlib

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from unitree_rl_lab.tasks.locomotion import mdp

from ...motion_dataset import MotionDatasetCfg
from ...reference_reset import reset_from_reference_motion
from .motion_cfg import (
    G1_AMP_KEY_LINK_NAMES,
    G1DanceMotionCfg,
    G1MixedMotionCfg,
    G1MotionSourceCfg,
    G1OmniRunMotionCfg,
    G1RunMotionCfg,
    G1WalkMotionCfg,
    G1WalkToRunMotionCfg,
    make_motion_dataset_cfg,
)


_velocity_cfg = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)
VelocityObservationsCfg = _velocity_cfg.ObservationsCfg
VelocityEventCfg = _velocity_cfg.EventCfg
VelocityCurriculumCfg = _velocity_cfg.CurriculumCfg
VelocityRobotEnvCfg = _velocity_cfg.RobotEnvCfg


def key_link_positions_in_base(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected body origins expressed in the robot root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    count = body_pos_w.shape[1]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, count, -1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, count, -1)
    positions_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    return positions_b.flatten(start_dim=1)


@configclass
class AMPObservationsCfg(VelocityObservationsCfg):
    @configclass
    class AMPCfg(ObsGroup):
        # TODO: 补全 AMP observation 的各个观测项。
        # 需要构造 x_amp in R^{B x H x d_amp} 中的单帧特征，例如：
        # base_lin_vel、base_ang_vel、projected_gravity、base_height、
        # joint_pos、joint_vel、key_links_pos_b。
        # 提示：可使用 ObsTerm(func=mdp.xxx)，关键连杆位置使用 key_link_positions_in_base。

        def __post_init__(self):
            # TODO: 设置 AMP observation group 的属性。
            # 提示：AMP observation 不应使用噪声扰动，并且需要将各观测项拼接成一个向量。
            self.enable_corruption = False

    amp: AMPCfg = AMPCfg()


@configclass
class AMPEventCfg(VelocityEventCfg):
    reset_reference = EventTerm(
        func=reset_from_reference_motion,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "probability": 0.5, "root_height_offset": 0.03},
    )


@configclass
class AMPCurriculumCfg(VelocityCurriculumCfg):
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


@configclass
class G1AMPFlatEnvCfg(VelocityRobotEnvCfg):
    """Common flat-ground AMP wiring; motion subclasses below contain task values."""

    observations: AMPObservationsCfg = AMPObservationsCfg()
    events: AMPEventCfg = AMPEventCfg()
    curriculum: AMPCurriculumCfg = AMPCurriculumCfg()
    motion_style: str = "mixed"
    motion_source: G1MotionSourceCfg = G1MixedMotionCfg()
    amp_history_steps: int = 3
    amp_motion: MotionDatasetCfg = make_motion_dataset_cfg(G1MixedMotionCfg())

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        self.amp_motion = make_motion_dataset_cfg(self.motion_source, self.amp_history_steps)
        self.amp_motion.step_dt = self.sim.dt * self.decimation
        self.events.reset_base = None
        self.events.reset_robot_joints = None


@configclass
class G1AMPWalkEnvCfg(G1AMPFlatEnvCfg):
    """Forward, backward, and turning walking."""

    motion_style: str = "walk"
    motion_source: G1WalkMotionCfg = G1WalkMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_reference.params["probability"] = 0.7

        self.rewards.track_lin_vel_xy.weight = 1.5
        self.rewards.track_ang_vel_z.weight = 0.5
        self.rewards.action_rate.weight = -0.03
        self.rewards.base_linear_velocity.weight = -2.0
        self.rewards.base_angular_velocity.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height.weight = -10.0
        self.rewards.gait.weight = 0.5
        self.rewards.feet_clearance.weight = 1.0
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.joint_deviation_waists.weight = -1.0
        self.rewards.joint_deviation_legs.weight = -1.0

        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.3)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class G1AMPRunEnvCfg(G1AMPWalkEnvCfg):
    """Pure steady forward running used as the AMP diagnostic baseline."""

    motion_style: str = "run"
    motion_source: G1RunMotionCfg = G1RunMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_lin_vel_xy.weight = 1.8
        self.rewards.base_linear_velocity = None
        self.rewards.base_angular_velocity = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.base_height = None
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (1.0, 4.2)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1AMPOmniRunEnvCfg(G1AMPRunEnvCfg):
    """Steady-run-dominant motion with commanded left and right turns."""

    motion_style: str = "omni_run"
    motion_source: G1OmniRunMotionCfg = G1OmniRunMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1AMPWalkToRunEnvCfg(G1AMPWalkEnvCfg):
    """Task-goal-driven reverse walk, walk, transition, run, and turning task."""

    motion_style: str = "walk_to_run"
    motion_source: G1WalkToRunMotionCfg = G1WalkToRunMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_reference.params["probability"] = 0.5
        self.events.reset_reference.params["synchronize_command"] = True
        self.rewards.track_lin_vel_xy.weight = 1.8
        self.rewards.action_rate.weight = -0.015
        self.rewards.joint_vel.weight = -0.0005
        self.rewards.joint_acc.weight = -1.25e-7
        self.rewards.feet_slide.weight = -0.1
        self.rewards.base_linear_velocity.weight = -0.5
        self.rewards.base_angular_velocity.weight = -0.02
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height.weight = -2.0
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 4.2)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1AMPDanceEnvCfg(G1AMPFlatEnvCfg):
    """Near-stationary Irish and Salsa motion."""

    motion_style: str = "dance"
    motion_source: G1DanceMotionCfg = G1DanceMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_reference.params["probability"] = 0.8
        self.rewards.track_lin_vel_xy.weight = 0.05
        self.rewards.track_ang_vel_z.weight = 0.05
        self.rewards.alive.weight = 0.12
        self.rewards.joint_acc.weight = -1.0e-7
        self.rewards.action_rate.weight = -0.01
        self.rewards.dof_pos_limits.weight = -0.5
        self.rewards.base_linear_velocity = None
        self.rewards.base_angular_velocity = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.base_height = None
        self.rewards.gait = None
        self.rewards.feet_clearance = None
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_waists = None
        self.rewards.joint_deviation_legs = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)
        self.terminations.bad_orientation.params["limit_angle"] = 1.31


@configclass
class G1AMPMixedEnvCfg(G1AMPFlatEnvCfg):
    motion_style: str = "mixed"
    motion_source: G1MixedMotionCfg = G1MixedMotionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 3.2)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-0.5, 0.5)


def _set_common_play_settings(cfg) -> None:
    cfg.scene.num_envs = 32
    cfg.scene.env_spacing = 2.5
    cfg.episode_length_s = 40.0
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class G1AMPWalkPlayEnvCfg(G1AMPWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)


@configclass
class G1AMPRunPlayEnvCfg(G1AMPRunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class G1AMPOmniRunPlayEnvCfg(G1AMPOmniRunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)


@configclass
class G1AMPWalkToRunPlayEnvCfg(G1AMPWalkToRunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)


# TODO: 补全 G1AMPWalkToRunFullPlayEnvCfg。
# 要求：继承 G1AMPWalkToRunPlayEnvCfg，关闭课程，设置完整评估命令范围，
# 例如支持低速走、高速跑和左右转弯。


@configclass
class G1AMPDancePlayEnvCfg(G1AMPDanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.05, 0.05)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1AMPMixedPlayEnvCfg(G1AMPMixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_common_play_settings(self)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)


@configclass
class G1AMPPlayEnvCfg(G1AMPMixedPlayEnvCfg):
    """Default Play configuration for the mixed task ID."""


# Reference-project compatibility aliases.
G1AmpFlatEnvCfg = G1AMPWalkEnvCfg
G1AmpFlatEnvCfg_run = G1AMPRunEnvCfg
G1AmpFlatEnvCfg_dance = G1AMPDanceEnvCfg
G1AmpFlatEnvCfg_PLAY = G1AMPPlayEnvCfg
