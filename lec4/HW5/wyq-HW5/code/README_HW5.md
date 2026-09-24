# 实践5 代码改动说明（按文件）

本文件说明 `code/` 目录中每个文件相对于课程原始作业包（unitree_rl_lab_student.zip）的改动。
目录结构保持与仓库相对路径一致。

## 一、Part 1 基础实现（7 个 TODO，3 个文件）

### 1. `source/unitree_rl_lab/unitree_rl_lab/tasks/navigation/mdp/observations.py`

- **TODO 1 `last_high_level_command`**：通过 `env.action_manager.get_term(action_name).processed_actions`
  返回裁剪后真正发给低层策略的高层速度指令（而非 raw_actions）。
- **TODO 2 `height_scan_pooled`**：调用 Isaac Lab `height_scan` 得到扁平射线高度后，用同文件
  `_height_scan_grid_shape` 按 ray pattern 恢复 (ny, nx) 二维网格，加通道维执行
  `F.max_pool2d(kernel=2, stride=2)` 再展平，将 1092 维高度扫描压缩为 273 维。

### 2. `source/unitree_rl_lab/unitree_rl_lab/tasks/navigation/mdp/pre_trained_policy_action.py`

- **TODO 3 `PreTrainedPolicyAction.__init__`**：
  - `read_file` + `torch.jit.load` 加载 TorchScript 低层策略，移到环境 device，
    置 `eval()` 且所有参数 `requires_grad_(False)`（冻结，无 optimizer）；
  - 将低层观测的 `velocity_commands` 绑定到裁剪后的高层动作 `_processed_actions`；
  - 将 `last_action`/`actions` 绑定到缓存的低层关节动作（episode 开始时清零）；
  - 创建只含 `ll_policy` 组的低层 `ObservationManager`。
- **TODO 4 `process_actions`**：完整保存输入到 `_raw_actions`（供日志/调试），
  按 `velocity_clip` 逐维 `torch.clamp` 到 `_processed_actions`，不原地修改输入。
- **TODO 5 `apply_actions`**（分层执行循环，`pre_trained_policy_action.py:100-111`）：
  - 仅当 `counter % low_level_decimation == 0` 时计算低层观测并在 `torch.inference_mode()`
    下执行冻结策略（低层 50 Hz，相邻推理间保持最近关节目标）；
  - **`compute_group("ll_policy", update_history=True)`**：必须更新历史，否则低层策略
    的 5 帧历史恒为"当前帧×5"（分布外输入），last_action 反馈环路发散，
    产生垃圾动作/观测导致机器人摔倒、critic 值爆炸、训练崩溃；
  - 低层 action term 的 `process_actions`/`apply_actions` 每个物理步都调用，并维护 counter。

### 3. `source/unitree_rl_lab/unitree_rl_lab/tasks/navigation/robots/g1/29dof/navigation_env_cfg.py`（Part 1 部分）

- **TODO 6 `make_low_level_inference_observations`**（`navigation_env_cfg.py:106-114`）：
  `deepcopy` 低层训练配置的 policy 观测组，关闭 corruption，保留 5 帧 history 与
  concatenate_terms，去掉 `base_ang_vel`/`projected_gravity`/`joint_pos_rel`/`joint_vel_rel`
  四项噪声，保证与 checkpoint 输入契约一致。
- **TODO 7 `NavigationObservationsCfg.PolicyCfg`**（`navigation_env_cfg.py:546-555`）：
  配置 `base_lin_vel`、`projected_gravity`、`base_ang_vel`(scale 0.2)、`pose_command`、
  `last_cmd`、`joint_pos_rel`、`joint_vel_rel`(scale 0.05)、`low_level_last_action`；
  corruption 关闭；高度扫描由 Compact 子类提供（273 维池化），critic-only 项不进 actor。
  最终 actor 观测 376 维。

## 二、Part 2 开放探索（方向 B：reset 时随机障碍生成）

### 4. `source/.../tasks/navigation/robots/g1/29dof/navigation_env_cfg.py`（Part 2 新增）

- **`V5R_DENSE_LEVELS = (0, 40, 80, 120)`**：自设的障碍数 curriculum 等级
  （区别于仓库 HRL-Extension 的 (0,50,80,100,120)），最终难度同为 120 个障碍。
- **`NavigationV5RandomDenseEventCfg`**：reset 事件，每次 episode 调用
  `randomize_mixed_obstacle_layout` 重新采样混合障碍布局（障碍类型/尺寸/场地范围与
  baseline 烘焙模板同分布，仅生成时机不同：reset 随机 vs startup 烘焙）；
  `reset_root_state_obstacle_aware` 保证机器人出生点避开障碍软约束区。
- **`NavigationV5RandomDenseCurriculumCfg`**：success-based 障碍数 curriculum，
  使用新函数 `obstacle_count_term_levels`，`level_counts=V5R_DENSE_LEVELS`。
- **`NavigationV5RandomDenseEnvCfg`**：继承 baseline 的 Compact_SingleGoal 配置
  （观测 376 维、单目标 5–10 m、奖励/终止/动作桥全部不变），仅替换 events 并加入
  curriculum——**障碍生成机制是唯一实验变量**。
- **`NavigationV5RandomDenseEnvCfg_PLAY`**：评估/演示用——俯视相机、每次 reset 固定
  120 障碍（`default_num_active=120`）、关闭 curriculum；并将 V5 为 4096 环境训练的
  固定大小 PhysX GPU buffer 缩小（`2^20/2^22/2^20`），使录像渲染在 12 GB 显卡上不 OOM。

### 5. `source/.../tasks/navigation/mdp/curriculums/obstacle_count.py`

- 新增 **`obstacle_count_term_levels`**：按 episode 终止原因升级的 curriculum。
  上次 episode 以 `goal_reached` 终止则升一级，跌倒（`base_height`/`bad_orientation`）
  降一级；写回 `env.obstacle_num_active` 供 reset 事件读取，返回 `level`/`num_active`
  指标供 TensorBoard 记录。
- **动机（发现的原实现缺陷）**：仓库自带 `obstacle_count_levels` 依赖命令项的
  `goals_reached` 指标判断升级，但该指标只在 `update_goal_on_success=True` 时累加
  （`ring_pose_command.py:187-191`）；本任务单目标命令到达目标即结束 episode，
  指标恒为 0，curriculum 永不升级。新函数直接读终止管理器，兼容单目标命令。
- 原 `obstacle_count_levels` 函数未做任何修改（HRL-Extension 不受影响）。

### 6. `source/.../tasks/navigation/robots/g1/29dof/__init__.py`

- 注册新任务 **`Unitree-G1-29dof-Navigation-HRL-RandomDense`**：
  训练入口 `NavigationV5RandomDenseEnvCfg`，play 入口 `NavigationV5RandomDenseEnvCfg_PLAY`，
  PPO 配置 `NavigationV5RandomDensePPORunnerCfg`。原有 Baseline/Extension 注册不变。

### 7. `source/.../tasks/navigation/robots/g1/29dof/agents/rsl_rl_ppo_cfg.py`

- 新增 **`NavigationV5RandomDensePPORunnerCfg`**：继承 baseline runner 全部超参
  （num_steps_per_env=8 等），仅 `experiment_name` 改为
  `unitree_g1_29dof_navigation_hrl_random_dense`，日志/checkpoint 写入独立目录。

### 8. `scripts/rsl_rl/evaluate.py`（新增）

- 统一评估脚本，用于 baseline 与 extension 的定量对比。固定协议：deterministic policy、
  seed 42、PLAY 入口（RandomDense 评估恒定 120 障碍、关闭 curriculum）、每个并行环境
  4 个 episode；按终止原因统计成功率/跌倒率/超时率/平均 episode 长度/末端目标误差，
  输出 JSON。
- 实现要点：episode 长度用自维护计数器（`episode_length_buf` 在 step 内已被 reset 清零）；
  终止分类在 `env.step` 返回后立即读取终止管理器 buffer（reset 不清除、下一次
  compute 才覆盖）；目标误差在 step 前快照（reset 后指标已对应新目标）。
- 用法：
  `python scripts/rsl_rl/evaluate.py --headless --task <TaskID> --checkpoint <model.pt> \
     --num_envs 64 --episodes_per_env 4 --output <summary.json>`

## 三、环境兼容性修复（非任务逻辑）

### 9. `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

- `UNITREE_MODEL_DIR` 改为环境变量覆盖：`os.environ.get("UNITREE_MODEL_DIR", ...)`，
  默认 `/home/star/projects/unitree_model`。在其它机器上运行前需
  `export UNITREE_MODEL_DIR=<unitree_model 路径>`。

### 10. `scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py`

- 对 `handle_deprecated_rsl_rl_cfg`、`get_published_pretrained_checkpoint` 的导入加
  try/except fallback：本机 Isaac Lab 2.3.1 未提供这些 helper（rsl-rl 3.x 不需要），
  不影响新版环境。

### 11. `unitree_rl_lab.sh`

- 仅追加注释：记录 1024 envs 下 Isaac Sim 5.1 / PhysX 的已知 GPU 死锁问题
  （512 envs 稳定），及 `CUDA_LAUNCH_BLOCKING=1` 可选缓解手段。无行为改动。

