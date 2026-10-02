# 实践 8：基于 AMP 的 G1 拟人走跑

根据《实践8：基于 AMP 的拟人走跑策略复现9.9更新版》补齐全部 9 项作业 TODO。

## 实现

| TODO | 实现位置 | 行为 |
| --- | --- | --- |
| 1 | `amp_flat_env_cfg.py` | 按专家数据顺序拼接基座线速度、角速度、投影重力、高度、绝对关节角、关节速度、双脚双手局部位置；80 维，不加噪声 |
| 2 | `agents/rsl_rl_ppo_cfg.py` | actor → policy，critic → critic，AMP → amp |
| 3–4 | `rsl_rl_amp/algorithms/discriminator.py` | `r_amp = dt * reward_scale * clamp(1 - (D - 1)^2 / 4, 0, 1)`；`r = (1 - alpha) * r_amp + alpha * r_task`；拒绝形状不匹配导致的广播 |
| 5 | `rsl_rl_amp/algorithms/amp.py` | 历史满 3 帧后使用混合奖励；此前保留完整任务奖励；终止后重新累计历史 |
| 6 | `rsl_rl_amp/algorithms/ppo.py` | `surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()` |
| 7 | `agents/rsl_rl_ppo_cfg.py` | 独立实验名 `unitree_g1_29dof_amp_walk_to_run`，profile 为 `walk_to_run`，任务权重 0.6、风格权重 0.4 |
| 8–9 | `amp_flat_env_cfg.py`、`config/g1/__init__.py` | FullPlay 继承走跑 Play；关闭课程、策略观测噪声、外力和推扰；注册独立 Gym 任务 |

环境配置位于 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp/config/g1/`。
单帧 80 维，三帧判别器窗口 240 维。关键连杆显式保序，关节使用仿真资产顺序；专家加载器按资产关节名称重排。

## 已接入的实践 7 数据

WalkToRun 默认读取 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp/data/hw7/`：

| 文件 | 动作 | 相对采样权重 |
| --- | --- | --- |
| `B3_-_walk1_stageii.npz` | 走路 | 4 |
| `B9_-_walk_turn_left_(90)_stageii.npz` | 左转 | 1 |
| `C3_-_Run_stageii.npz` | 跑步 | 5 |
| `C5_-_walk_to_run_stageii.npz` | 走转跑 | 3 |

环境、Runner、数据集的 profile 均为 `walk_to_run`。保留已有的独立 `hw7` profile。
可通过 `UNITREE_AMP_WALK_TO_RUN_DIR` 指定包含相同文件名的其他目录；或通过 `UNITREE_AMP_MOTION_ROOT` 指定数据根目录（其下应有 `hw7/`）。

目前没有右转、倒退和跑转停的独立专家示范。代码可使用这四条数据开始训练，但不能据此保证完整验收效果。后续补齐动作时，在 `G1WalkToRunMotionCfg.clip_weights` 中加入真实文件名及正权重；加载器会检查引用的文件是否存在。Walk、Run、OmniRun、Mixed、Dance 仍使用各自原始数据配置，启动这些任务前需准备其对应数据。

## CPU 验证

在本目录、具有项目依赖的 Python 环境中执行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

测试不依赖 Isaac Sim，使用真实 HW7 数据和真实 PyTorch/TensorDict 算法。覆盖：

- LSQ 奖励边界、步长缩放、混合权重端点及形状检查。
- 四条 NPZ 的 80/240 维特征、有限值、关节名称重排、参考状态采样。
- 终止帧与下一 episode 重置帧分离。
- 不足三帧时保留任务奖励、有效窗口进入 replay buffer、终止后重新累计。
- 一次完整 rollout → GAE → PPO 更新 → 判别器更新，参数变化及有限梯度。

本次本机验证：5 项测试全部通过。测试借用了 HW6 的 CPU Python 环境，并将缺少的 Gymnasium 安装在临时目录，未改动 HW6/HW7 环境。

## Isaac Lab 训练与评估

以下命令在本目录、已安装 Isaac Sim/Lab 的训练环境中执行。本机没有 Isaac Lab，以下仿真步骤尚未运行，未生成训练 checkpoint、曲线或效果视频。

```bash
conda activate env_isaaclab
python -m pip install -e .
python -m pip install -e source/unitree_rl_lab
python scripts/list_envs.py

# 先验证实际仿真观测、专家数据、配置和一步环境交互
python scripts/check_hw8_isaac.py --headless --num_envs 4

# 短训练检查，再启动正式训练
python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-AMP-WalkToRun --num_envs 64 --max_iterations 2
python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-AMP-WalkToRun --num_envs 4096
```

显存不足时将 `--num_envs` 降为 1024。日志位于 `logs/rsl_rl_amp/unitree_g1_29dof_amp_walk_to_run/<时间戳>/`。

```bash
tensorboard --logdir logs/rsl_rl_amp/unitree_g1_29dof_amp_walk_to_run

python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-AMP-WalkToRun-FullPlay \
  --checkpoint /path/to/model.pt --num_envs 32

python scripts/rsl_rl/play.py --headless \
  --task Unitree-G1-29dof-AMP-WalkToRun-FullPlay \
  --checkpoint /path/to/model.pt --num_envs 1 --video --video_length 2500
```

默认控制步长 0.02 秒，2500 步约 50 秒。Play 会自动导出 `exported/policy.pt` 和 `exported/policy.onnx`。

FullPlay 的前后速度范围为 −1.0～4.2 m/s，横向速度为 0，角速度范围为 −1.2～1.2 rad/s。它按原命令生成器随机采样，单个短视频不保证覆盖每个验收场景。

观察 task/style/mixed reward、PPO surrogate/value loss、entropy、判别器 loss、expert/policy score、valid window fraction，以及线速度/角速度跟踪。实际检查低速走、高速跑、减速、左右转弯、摔倒率、滑步和摆臂。正式训练成功与否须以这些曲线及仿真表现为准。
