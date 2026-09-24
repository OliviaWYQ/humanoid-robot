# 实践6实验报告：基于教师-学生蒸馏的全身运动跟踪

## 1. 实验设置

- 框架：mjlab 1.2.0 + Unitree G1，PPO（`num_steps_per_env=24`，`max_iterations=20000`）
- 数据：AMASS/ACCAD 子集 25 个 walk clips（已完成 retargeting，以 motion command 提供）
- 教师：预训练 privileged teacher（`checkpoints/g1_hw6_teacher/model_latest.pt`），21 步 future reference command + proprio + 10 步 history，共 3007 维观测；训练时冻结，仅提供监督信号
- 学生：只能观测当前参考帧 + proprioception + history，接近可部署场景
- 硬件：RTX 4070 12 GB，按作业 §3.6 显存表使用 1024 并行环境
- 总 loss：`loss = surrogate + value_loss_coef·value − entropy_coef·entropy + distill_coef·distill_loss`，蒸馏系数随训练线性退火

## 2. Teacher-Student Distillation 原理简述

Motion tracking 的目标是让机器人跟踪参考运动。教师策略训练时可以使用**特权信息**——当前帧 + 未来 20 步的参考运动（21×64 维 command block）。提前"看到"未来轨迹使教师能做出更稳定、更 anticipatory 的跟踪动作，因此单个教师即可联合跟踪全部 25 个 clip。但未来参考帧在真实部署中不可获得，教师无法直接部署。

蒸馏的核心思想：用冻结的教师作为监督信号，训练只能看到当前帧的学生策略，把教师的"跟踪能力"迁移到可部署的观测条件下。实现是在标准 PPO 损失上加一项蒸馏损失，并用线性退火的 `distill_coef` 平衡模仿与强化：

- 训练早期系数大，学生快速模仿教师、获得较好的初始步态；
- 训练后期系数小，PPO 的 tracking reward 主导，学生在自己的观测条件下自主优化，避免被教师"带偏"。

本作业实现了两种蒸馏匹配方式：

**Action Matching**：回归教师在同一 batch 上输出的动作（MSE/Huber），即"老师此刻会怎么做"。实现简单、训练稳定，作为 baseline。

**KL Matching**：匹配教师与学生动作分布的 KL(teacher‖student)（对角高斯，同时约束 mean 和 std），即复制"老师对动作的完整概率分布"。信息更丰富，可以把教师的不确定性结构也传递给学生。

## 3. Action Matching vs KL Matching 对比

### 3.1 训练稳定性

| 指标 | Action Matching（首 → 终） | KL Matching（首 → 终） |
|---|---|---|
| 蒸馏主 loss | `Loss/bc`: 0.502 → 0.033 | `Loss/kl`: 45.6 → 1.39 |
| 诊断指标 | `action_mae`: 0.542 → 0.136 | `mean_rmse`: 0.624 → 0.063；`std_rmse`: 0.206 → 0.010 |
| `distill_coef` 退火 | 1.00 → 0.05（全程退火到位） | 0.100 → 0.070（配置退火周期 60k > 实际 20k 迭代，未退到 0.01） |
| `Train/mean_reward` | 0.64 → 18.44（@10k 18.5，@15k 20.3） | 0.63 → 22.06（@10k 21.6，@15k 21.6） |
| `Train/mean_episode_length` | 22.6 → 184.5 | 22.6 → 213.2 |
| `Termination_Frac/fail` | ≈0.99（早期）→ 0.000 | ≈0.99（早期）→ 0.000 |

分析：

1. **收敛速度**：KL Matching 明显更快。其 reward 在约 10k 迭代即收敛到 ~21.6 且之后基本平稳；Action Matching 到 15k 才到 ~20.3，且后期（15k→20k）reward 有轻微回落（20.3→18.4），同时 episode length 从 205 降到 184，说明退火后期 reward 主导阶段其策略有小幅退化。
2. **loss 形态**：两种蒸馏 loss 都单调下降、无震荡，说明两种集成方式都稳定。Action Matching 的 `action_mae≈0.14`（动作幅值量级约 1），即学生动作与教师动作的偏差已较小；KL Matching 的 `mean_rmse≈0.06` 更小，且 `std_rmse≈0.01` 表明学生还学会了教师动作分布的宽度（不确定性结构），这是 action matching 无法传递的信息。
3. **退火差异**：Action Matching 的系数从 1.0 退到 0.05，后期 PPO 主导更充分；KL 系数只退到 0.07，教师监督在整个训练中保持较强——这与其"全程稳定、几乎无后期退化"的曲线一致，也提示 KL 方法对退火 schedule 的敏感性较低。

### 3.2 Tracking 质量（Play 验证）

Viser 回放中两种方法均能稳定跟踪 25 个 walk clips，全程不跌倒、步态自然。定量上（`motion_stats_latest.csv`，25 clips 平均）：

| 指标 | Action Matching | KL Matching |
|---|---|---|
| `failure_ema` 均值 | 0.00057 | 0.00032 |
| `completion_ema` 均值 | 0.9954 | 0.9973 |
| 平均 episode 终止方式 | `time_out` ≈ 1.0（走完整段 clip） | `time_out` = 1.0（走完整段 clip） |

两类方法的难点 clip 一致：**转身类动作**（`Walk_turn_right_90`、`Walk_turn_around`、`walk_backwards` 等）failure_ema 最高、completion 最低，因为根部大角度转动对全身协调要求最高；直线行走 clip 基本 completion≈1.0。KL Matching 在最难的 clip 上仍保持更低失败率（其最差 clip `Walk_backwards` failure_ema≈0.0015，对应 Action Matching 最差 clip `Walk_turn_right_90` failure_ema≈0.0028），整体 tracking 质量略优。

### 3.3 Tracking 相关指标及其含义

评估 motion tracking 质量主要看四类指标：

1. **任务层面**：`Train/mean_reward`（tracking 各 reward 项之和）、`Train/mean_episode_length`（越长说明越不容易提前失败）、`Termination_Frac/fail` vs `time_out`（理想状态是 fail→0、time_out→1，即 episode 都以"走完整段参考动作"结束而非摔倒/超时）。
2. **跟踪误差层面**：`Metrics/motion/error_anchor_pos/rot/lin_vel/ang_vel`（骨盆 anchor 的位置/姿态/线速度/角速度跟踪误差）、`error_body_pos/rot`、`error_joint_pos/vel`，直接度量机器人与参考动作的逐帧偏差，是 tracking 质量最直接的指标，数值越小越好。
3. **逐 clip 统计**：`motion_stats/motion_stats_latest.csv` 中的 `failure_ema`、`completion_ema`、`attempts`、`sampling_ratio`——前者给出每条参考动作上的稳定失败率与完成度，用于发现方法的短板 clip。
4. **蒸馏层面（诊断用）**：`Loss/bc`、`Loss/kl` 是训练信号本身；`action_mae/rmse`（action matching）与 `mean_rmse/std_rmse`（KL matching）度量学生与教师动作（分布）的距离，反映模仿的保真度；`Loss/distill_coef` 记录退火过程。这些是训练诊断指标，不直接等于 tracking 质量，但解释了 tracking 差异的来源。

### 3.4 结论

- 两种蒸馏方法都成功把教师能力迁移到了可部署观测下，25 个 clip 零失败跟踪。
- **KL Matching 整体更好**：收敛更快（10k vs 15k）、最终 reward 更高（22.1 vs 18.4）、episode 更长、逐 clip 失败率更低；并且额外传递了教师动作分布的 std 信息，训练全程更平稳、无后期退化。
- **Action Matching 更简单稳定**，作为 baseline 已足够好用，适合快速验证流程；其后期 reward 小幅回落说明退火末端纯 reward 主导阶段对 motion tracking 这种精细任务略有风险。
- 适用场景：算力/时间受限、追求实现简单 → action matching；追求最终 tracking 质量与训练稳定性 → KL matching。

## 4. 附图（建议）

- `figures/tb_action_matching.png` / `figures/tb_kl_matching.png`：TensorBoard 截图（mean_reward、mean_episode_length、Termination_Frac/fail、Loss/bc 与 Loss/kl、诊断指标、distill_coef）
- `figures/viser_*.mp4/gif`：两种 student 的 Viser 回放录屏
