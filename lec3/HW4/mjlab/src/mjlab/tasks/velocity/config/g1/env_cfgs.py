"""Unitree G1 velocity environment configurations.

Homework TODOs in this file: **3**, **4**, **5**  (of 10 total)
Function: unitree_g1_flat_height_env_cfg()
Index: docs/HOMEWORK_TODO.md · grep: 【作业 TODO
"""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import (
  UniformBaseHeightCommandCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_base_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  velocity_cmd = cfg.commands["velocity"]
  assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
  velocity_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

    velocity_cmd = cfg.commands["velocity"]
    assert isinstance(velocity_cmd, UniformVelocityCommandCfg)
    velocity_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    velocity_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  return unitree_g1_base_env_cfg(play=play)


def unitree_g1_flat_height_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity + height command configuration."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # >>> HOMEWORK_TODO_3_START
  # ==============================================================================
  # 【作业 TODO 3/10】注册 base_height 命令
  # 位置: config/g1/env_cfgs.py · unitree_g1_flat_height_env_cfg
  # 提示: 高度指令是 height 任务的输入；entity_name 应对应 scene 中的 robot。
  # 概念: UniformBaseHeightCommandCfg、resampling_time_range、ranges.height
  # 索引: docs/HOMEWORK_TODO.md · 完整说明见 docs/HW3_蹲姿行走策略.md
  # ==============================================================================
  # --- 实现提示 ---
  # - 向 cfg.commands 注册 "base_height"
  # - 参考 velocity_env_cfg.py 中 commands["velocity"] 的写法
  # - 需配置 entity_name、resampling_time_range、ranges.height（见 HW3 §2.2 / §6）
  # <<< HOMEWORK_TODO_3_END

  # >>> HOMEWORK_TODO_4_START
  # ==============================================================================
  # 【作业 TODO 4/10】接入 height_command 观测（蹲姿/高度任务核心观测）
  # 位置: config/g1/env_cfgs.py · unitree_g1_flat_height_env_cfg
  # 提示: 策略必须能观测到 base_height 指令；command_name 与 commands 键一致。
  # 概念: ObservationTermCfg、generated_commands、actor/critic 观测拼接
  # 索引: docs/HOMEWORK_TODO.md · 完整说明见 docs/HW3_蹲姿行走策略.md
  # ==============================================================================
  height_command_obs = ObservationTermCfg(
    func=envs_mdp.generated_commands,
    params={"command_name": "TODO"},  # TODO 4: 替换为正确的 command 名称
  )
  # --- 实现提示 ---
  # - params["command_name"] 须与 TODO 3 注册的 commands 键一致
  # - 将 height_command 观测项加入 cfg.observations["actor"] 与 ["critic"] 的 terms
  # <<< HOMEWORK_TODO_4_END

  # >>> HOMEWORK_TODO_5_START
  # ==============================================================================
  # 【作业 TODO 5/10】注册 track_base_height 奖励项（蹲姿/高度任务核心奖励）
  # 位置: config/g1/env_cfgs.py · unitree_g1_flat_height_env_cfg
  # 提示: 将 track_base_height 接入奖励管理器；weight/std 已预填，无需修改。
  # 概念: RewardTermCfg、command_name="base_height"
  # 索引: docs/HOMEWORK_TODO.md · 完整说明见 docs/HW3_蹲姿行走策略.md
  # ==============================================================================
  # --- 实现提示 ---
  # - 向 cfg.rewards 注册 "track_base_height"
  # - func 指向 mdp.track_base_height；params 含 command_name 与 std（见 HW3 §6 TODO 5）
  # <<< HOMEWORK_TODO_5_END

  return cfg
