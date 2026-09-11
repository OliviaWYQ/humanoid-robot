# 实践 4：蹲姿行走 TODO 实现说明

依据 `lec3/实践4：蹲姿行走策略，G1速度+骨盆高度MDP设计.pdf` 第 2、6 节完成。
以下路径相对于 `mjlab/`；源码保留 `HOMEWORK_TODO_N_START/END` 标记，便于对照作业。

| TODO | 文件（`src/mjlab/tasks/velocity/` 下） | 位置与实现 |
| --- | --- | --- |
| 1 | `mdp/height_command.py` | `_resample_command`：仅对 `env_ids` 在配置范围内向量化均匀采样 |
| 2 | `mdp/height_command.py` | `_update_metrics`：累积绝对高度误差 / 最长指令周期步数，记录目标高度 |
| 3 | `config/g1/env_cfgs.py` | `unitree_g1_flat_height_env_cfg`：注册 `base_height`，范围 0.45–0.80 m |
| 4 | `config/g1/env_cfgs.py` | Actor、Critic 均添加 `height_command`，读取 `base_height` |
| 5 | `config/g1/env_cfgs.py` | 注册 `track_base_height`，权重 1.0 |
| 6 | `mdp/rewards.py` | `track_base_height`：世界坐标骨盆 z 的平方误差高斯奖励 |
| 7 | `mdp/rewards.py` | `track_linear_velocity`：机体坐标 xy 跟踪误差 + z 速度平方 |
| 8 | `mdp/rewards.py` | `track_angular_velocity`：yaw 跟踪误差 + roll/pitch 角速度平方 |
| 9 | `velocity_env_cfg.py` | `curriculum["command_vel"]`：0、120000、240000 环境步三阶段 |
| 10 | `mdp/observations.py` | `foot_contact`：`(found > 0).float()`，仅 Critic 可见 |

## 配置选择

- 所有跟踪奖励均为 `exp(-平方误差 / std**2)`，输出 `[num_envs]`。
- 高度是世界坐标绝对 z；本任务在平地上运行，不扣除地形或环境原点高度。
- 高度指令重采样间隔沿用速度命令的 `(3.0, 8.0)` 秒，独立采样，开启调试球体。
- 文档说高度奖励 `std` 已预填，但提供的代码中该配置整段缺失，PDF 也未给出数值。
  因此采用 **`std=0.1 m` 的实现假设**：误差 10 cm 时奖励约 0.368。
  高度权重 1.0 按 PDF 给定值；其他已有权重和 std 保持原预设。
- `target_height_mean` 用 `copy_` 写入独立指标缓冲区，数值符合文档；避免直接引用
  `height_command[:, 0]` 后，CommandManager 清空指标时连带修改指令。
- Stage 2 不写 `ang_vel_z`，课程函数按顺序应用各阶段，保留 Stage 1 的 ±0.7 rad/s。

## 奖励设计

源码中已逐项标注类别和意图：

| 类别 | 奖励 | 作用及权重依据 |
| --- | --- | --- |
| Task | linear / angular velocity（各 2.0），base height（1.0） | 速度项提供行走和转向能力，高度项叠加蹲姿目标；使用文档权重 |
| Style | upright（1.0），pose（1.0），air_time（0.0） | 保持躯干姿态和步态；G1 的腾空项默认关闭，保留用于消融 |
| Reg | action_rate_l2（-0.1），soft_landing（-1e-5） | 平滑动作并减少落地冲击；接触力量纲大，权重较小 |
| Penalty | dof_pos_limits（-1.0），foot_clearance（-2.0），foot_swing_height（-0.25），foot_slip（-0.1） | 关节软限位、足部离地和滑移约束 |
| Penalty | body_ang_vel（-0.05），angular_momentum（-0.02），self_collisions（-1.0） | 减少躯干晃动、整体旋转和自碰撞 |

这些解释对应现有配置，不代表已经通过训练验证其最优性。

## 验证范围

`tests/test_velocity_task.py` 检查任务注册和基础配置。
新增 `tests/test_velocity_height_mdp.py` 使用真实 PyTorch 张量验证采样范围、非目标环境
保持不变、空索引、指标归一化与缓冲区隔离、奖励数值、未指令轴惩罚、特权观测，
以及课程切换前后边界。这些测试不需要 GPU。

完整作业的训练曲线、至少两组消融和 `h=0.5 m` 连续蹲走 ≥10 秒，需要训练后另行验证。
代码与短程运行检查不能作为策略收敛或蹲走达标的证据。

### 2026-09-05 macOS 实测结果

- Apple Silicon / macOS 26.6.2 / Python 3.13.12，Warp 设备为 `cpu`。
- `test_velocity_task.py` + `test_velocity_height_mdp.py`：**21 passed**。
- 修改的 5 个源码文件和新增测试、检查脚本：Ruff 检查及格式检查通过。
- `list_envs` 正确列出 G1 速度任务和速度 + 高度任务。
- 单环境 CPU 检查通过：reset + 2 步仿真，Actor `[1, 100]`、Critic `[1, 112]`；
  高度指令约 0.5057 m，观测和奖励均为有限值。含首次 Warp 编译约 6.67 秒。
- CPU **2 个并行环境 × 24 步 × 1 次 PPO 迭代**成功完成，共 48 个 transition。
  本次日志记录采样约 0.434 秒、更新约 0.043 秒（小样本冒烟结果，不应外推大规模性能）。
  日志及模型保存在 `/tmp/hw4-training-logs/g1_velocity_height/`。
- 因此这台 Mac 可以运行该任务的 CPU 仿真与小规模 PPO；没有完成策略收敛训练、
  Viser 交互验收或 0.5 m 蹲走性能测试。文档的 4096 环境完整训练仍建议使用 Linux + NVIDIA GPU。

官方平台说明：[mjlab FAQ](https://mujocolab.github.io/mjlab/v1.1.1/source/faq.html)、
[Warp 平台支持](https://nvidia.github.io/warp/latest/user_guide/compatibility.html)。

## 2026-09-11：修复标准 uv 启动

之前通过直接调用 `.venv/bin/python` 验证了 CPU 运行，但没有修改旧依赖锁。
`uv run` 会先同步项目环境，因此仍会请求旧 nightly wheel 并报 404。
本次将项目依赖固定为 MuJoCo 3.8.1 / MuJoCo Warp 3.9.0.1 正式版，移除 MuJoCo
nightly 源与 Warp Git 源，并重新生成 `uv.lock`。
通用 Python 包使用清华 PyPI 镜像，CUDA 专用包仍使用原官方索引。
锁文件的大部分差异是下载地址变化，除 MuJoCo 从 nightly 改成正式版外，
其他包的锁定版本未升级。

同时补齐原压缩包缺失的 README 和 Apache-2.0 LICENSE，使项目能够安装并提供
`train` / `play` 命令。没有 CUDA 设备且未设置 `CUDA_VISIBLE_DEVICES` 时，默认
GPU 选择现会返回 CPU，避免 `gpu_ids=[0]` 索引空列表。
修复 `list-envs` 控制台入口将环境数量作为退出码的问题，正常列举任务返回 0。

本次验证：`uv lock --check --offline` 通过；通过 `uv run --locked` 执行的
34 项任务/GPU 选择测试通过；标准 `uv run --locked train` 在未指定 `--gpu-ids`
时自动选择 CPU，并完成 2 个环境、24 步、1 次 PPO 迭代（48 个 transition）。
该次日志位于 `/tmp/hw4-uv-training-logs/g1_velocity_height/`。

## 本地运行命令

以下命令均在 `lec3/HW4/mjlab/` 中执行，不需要手动设置 `PYTHONPATH`。

```bash
uv sync --locked --extra cpu
uv run list-envs

# macOS CPU：先用 2 个环境、1 次 PPO 迭代验证完整训练链路
uv run train Mjlab-VelocityHeight-Flat-Unitree-G1 \
  --gpu-ids None --env.scene.num-envs 2 \
  --agent.max-iterations 1 --agent.logger tensorboard

# 单元测试与无窗口仿真检查
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_velocity_task.py tests/test_velocity_height_mdp.py tests/test_gpu_selection.py -q
uv run python scripts/check_velocity_height.py
```

`--agent.logger tensorboard` 将训练日志保存在本地，避免默认 W&B 登录流程。
1 次迭代仅检查训练链路；增加 `--agent.max-iterations` 才会继续训练。
文档中的 4096 环境是 GPU 并行训练规模，不建议直接在 Mac CPU 上使用。

Linux + NVIDIA GPU 的正式训练：

```bash
uv sync --locked --extra cu128
uv run --extra cu128 train Mjlab-VelocityHeight-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 --agent.logger tensorboard
```

若下载出现连接超时，检查终端代理设置。系统代理不一定被 `uv` 自动使用；
本机检查到的 HTTP/HTTPS 代理地址为 `127.0.0.1:7897`，该代理服务开启时可设置：

```bash
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
```

代理只能解决连接问题，无法修复已删除资源的 HTTP 404；404 由本次依赖更新解决。

在受限环境运行时，可将缓存指定到可写目录：

```bash
export WARP_CACHE_PATH=/tmp/hw4-warp-cache
export MPLCONFIGDIR=/tmp/hw4-matplotlib
export PYTHONPYCACHEPREFIX=/tmp/hw4-pycache
```
