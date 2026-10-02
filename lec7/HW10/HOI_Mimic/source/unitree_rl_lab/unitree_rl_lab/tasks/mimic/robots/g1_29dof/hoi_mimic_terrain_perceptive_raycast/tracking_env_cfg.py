from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.hoi_mimic_terrain import tracking_env_cfg as blind_cfg
from unitree_rl_lab.tasks.mimic.sensors import HoiMergedTerrainRayCasterCfg


@configclass
class RobotSceneCfg(blind_cfg.RobotSceneCfg):
    # TODO(student): replace None with a HoiMergedTerrainRayCasterCfg.
    # Configure the torso-mounted, yaw-aligned grid scanner described in
    # HOI_MIMIC_HOMEWORK.md. Keep the supplied RayCaster implementation unchanged.
    height_scanner: HoiMergedTerrainRayCasterCfg | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        # TODO(student): replace None with the policy height_scanner ObsTerm.
        # The policy term uses mdp.height_scan, clipping, history, and uniform noise.
        height_scanner: ObsTerm | None = None
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=blind_cfg.PROPRIO_HISTORY_LENGTH
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=blind_cfg.PROPRIO_HISTORY_LENGTH
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        # TODO(student): replace None with the critic height_scanner ObsTerm.
        # Match the policy height semantics and history, but do not add noise.
        height_scanner: ObsTerm | None = None
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        actions = ObsTerm(func=mdp.last_action, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


def _check_homework_todos(cfg) -> None:
    """Fail with a focused message until the three assignment TODOs are complete."""
    if cfg.scene.height_scanner is None:
        raise NotImplementedError(
            "TODO(student): configure RobotSceneCfg.height_scanner as described in HOI_MIMIC_HOMEWORK.md."
        )
    if cfg.observations.policy.height_scanner is None:
        raise NotImplementedError(
            "TODO(student): add the policy height_scanner observation described in HOI_MIMIC_HOMEWORK.md."
        )
    if cfg.observations.critic.height_scanner is None:
        raise NotImplementedError(
            "TODO(student): add the critic height_scanner observation described in HOI_MIMIC_HOMEWORK.md."
        )


@configclass
class RobotEnvCfg(blind_cfg.RobotEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _check_homework_todos(self)
        self.scene.height_scanner.env_spacing = float(self.scene.env_spacing)


@configclass
class RobotPlayEnvCfg(blind_cfg.RobotPlayEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _check_homework_todos(self)
        self.scene.height_scanner.env_spacing = float(self.scene.env_spacing)
