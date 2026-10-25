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

本次本机验证：5 项测试全部通过（`env_isaaclab`，Python 3.11，torch 2.7.0+cu128、tensordict 0.13.0、Gymnasium 1.2.0 均已就绪，无需额外安装）。

## Isaac Lab 训练与评估

本机环境：Ubuntu 22.04，2× RTX 4070 12 GB，IsaacLab 0.48.0 + Isaac Sim 5.1.0（kit 安装位于 `/home/star/isaacsim`，经 `IsaacLab/_isaac_sim` 软链接接入）。`conda activate env_isaaclab` 的激活脚本会自动加载 Isaac Sim 的 Python 路径。

机器适配（已配置在 `env_isaaclab/etc/conda/activate.d/setenv.sh`）：

- `UNITREE_MODEL_DIR=/home/star/projects/unitree_model`——机器人 USD 的默认路径 `~/unitree_rl_lab-main/unitree_model` 在本机不存在，不设置会在建环境时报 `FileNotFoundError`。

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

# 正式训练（推荐本机用法）：单卡 + 崩溃自动重启/断点续训。
# 包装脚本内含 300 秒无进展看门狗（比 NCCL 600 秒看门狗更快止损），
# 崩溃后自动从最新 checkpoint 续训。
bash scripts/train_autoresume.sh

# 等价的裸命令（单卡 1024 环境，30000 次迭代）
python scripts/rsl_rl/train.py --headless \
  --task Unitree-G1-29dof-AMP-WalkToRun --num_envs 1024
```

本机 12 GB 显存无法容纳 4096 环境。最终采用单卡 1024 环境：实测约 0.77 s/迭代、约 3.8 GB 显存，30000 次迭代预计约 6～6.5 小时。日志位于 `logs/rsl_rl_amp/unitree_g1_29dof_amp_walk_to_run/<时间戳>/`。

实测教训（本机，Isaac Sim 5.1 / PhysX 107.3）：

- **双卡分布式在本机不可用**：无论每卡 1024 还是 512 环境，训练都在 1～2 分钟内停滞（NCCL 集体通信双 rank 阻塞、GPU 100% 空转）；`CUDA_LAUNCH_BLOCKING=1` 无法避免且拖慢 55%。死锁随分布式模式出现而非随环境数出现（单卡 1024 环境稳定）。两卡 P2P 仅 5.6 GB/s（卡 1 为 PCIe x4），即便无死锁也仅与单卡同速。如需恢复双卡，改 `scripts/train_autoresume.sh` 顶部的 `NUM_GPUS=2` 并自行验证稳定性。
- 本机 `list_envs.py` 会自启一个 Kit 实例，与正在进行的训练互斥（Kit 键值数据库锁），需在训练停止后运行。

本机已验证：

- `check_hw8_isaac.py` 通过（profile、FullPlay 配置、80/240 维特征、专家数据、4 步环境交互全部断言成立）。
- 2 次迭代短训练通过：AMP 判别器 loss、policy/expert score、style reward 等指标正常，checkpoint 正常保存。
- 正式训练已启动并运行中（单卡 1024 环境、30000 次迭代，经 `scripts/train_autoresume.sh` 包装）。

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
