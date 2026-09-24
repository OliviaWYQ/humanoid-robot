# 实践5实验报告：基于分层强化学习的人形机器人导航

## 1. 分层控制结构与控制频率

本作业实现两层分层导航系统：高层 PPO 导航策略根据目标、局部高度感知和本体状态输出
3 维速度指令 `(vx, vy, yaw_rate)`；冻结的低层 locomotion 策略（Unitree-G1-29dof-LowLevel
预训练权重）将速度指令转换为 29 维关节位置目标。

三个层级使用不同时间尺度：

| 层级 | 周期 | 频率 | 说明 |
|---|---|---|---|
| 物理仿真 | 0.005 s | 200 Hz | `sim.dt` |
| 低层策略 | 0.02 s | 50 Hz | `low_level_decimation=4`，两次推理间保持最近关节目标 |
| 高层策略 | 0.2 s | 5 Hz | `decimation=4×10=40` |

层间接口为裁剪到 `vx∈[-0.5,1.0]`、`vy∈[-0.5,0.5]`、`yaw_rate∈[-0.5,0.5]` 的速度指令，
经 `PreTrainedPolicyAction` 写入低层观测的 `velocity_commands` 项。速度指令比 29 维关节
动作更紧凑、可解释：可直接检查高层"要求机器人往哪走"、命令是否饱和，并能区分失败来自
规划层还是运动层。

## 2. Part 1：高层观测设计（376 维）

高层 actor 观测由"目标 + 局部感知 + 本体状态 + 控制历史"拼接：

| Term | 维度 | 说明 |
|---|---|---|
| base_lin_vel | 3 | 机体线速度 |
| projected_gravity | 3 | 姿态/倾斜 |
| base_ang_vel (×0.2) | 3 | 机体角速度 |
| pose_command | 4 | 机器人坐标系下的目标位置与朝向 |
| last_cmd | 3 | 上一条裁剪后的高层指令 |
| height_scan_pooled | 273 | 2×2 max-pool 后的局部高度扫描（1092→273） |
| joint_pos_rel | 29 | 相对关节位置 |
| joint_vel_rel (×0.05) | 29 | 相对关节速度 |
| low_level_last_action | 29 | 冻结低层策略上一关节动作 |
| **合计** | **376** | |

Critic 额外观察 base height 与 command distance（不入 actor）。高度扫描池化在保留局部
障碍空间结构的同时将网络输入压缩 4 倍。

## 3. 冻结低层策略的加载与连接

`PreTrainedPolicyAction`（`pre_trained_policy_action.py`）负责桥接：

1. `read_file` + `torch.jit.load` 加载 TorchScript 权重，移到环境 device，`eval()` 且
   `requires_grad_(False)`，全程无 optimizer、不反向传播；
2. 低层观测 `deepcopy` 自低层训练配置（5 帧 history、关闭 corruption 与四项噪声），
   `velocity_commands` 绑定到裁剪后的高层动作，`last_action` 绑定到缓存的低层动作；
3. 执行循环：仅当 `counter % 4 == 0` 时计算低层观测并在 `torch.inference_mode()` 下推理，
   其间必须 `update_history=True`（否则 5 帧历史恒为当前帧复制，分布外输入导致
   last_action 反馈环路发散、训练崩溃）；低层 action term 每个物理步都执行。

## 4. Baseline 训练配置与结果

- 完整训练（Part 1）：`num_envs=1024, max_iterations=30000, seed=42`（约 246M env steps）。
  最终迭代：goal_reached 终止比例 **0.985**，base_height 0.000，bad_orientation 0.000，
  time_out 0.015，目标误差 0.45 m。
- 对照训练（Part 2 协议）：`num_envs=512, max_iterations=10000, seed=42`（约 41M env steps，
  与 extension 严格同预算）。最终迭代：goal_reached **0.967**，跌倒 0.000，
  time_out 0.033。1024 envs 时 PhysX 存在已知 GPU 死锁问题，故对照实验采用 512 envs。

## 5. Part 2 开放探索（方向 B：reset 时随机障碍生成）

**问题定义**：baseline 的三张地图在环境启动时烘焙、episode 间保持不变，策略可能过拟合
有限模板。假设：reset 时随机生成布局训练出的策略具有更强泛化能力。

**实现**（唯一实验变量 = 障碍生成机制）：注册新任务 `Unitree-G1-29dof-Navigation-HRL-RandomDense`，
严格继承 baseline 全部配置（376 维观测、单目标 5–10 m、奖励/终止/动作桥不变），仅将
startup 烘焙事件替换为 `randomize_mixed_obstacle_layout`（reset 时重采样），并配套
success-based 障碍数 curriculum：等级 (0, 40, 80, 120)，episode 以 goal_reached 终止则
升一级、跌倒降一级（新实现 `obstacle_count_term_levels`）。

**发现并修复的原实现缺陷**：仓库自带 `obstacle_count_levels` 依赖命令项 `goals_reached`
指标升级，但该指标仅在 `update_goal_on_success=True` 时累加；单目标命令到达目标即结束
episode，指标恒为 0，curriculum 永不升级（我们第一版训练因此全程 0 障碍，重训修复）。
新函数直接按终止管理器的终止原因升级，兼容单目标命令。

**难度设置**：最终评估恒定 120 个混合障碍（圆柱半径 0.25/0.40/0.55 m + 低/高方墩，
最小中心间距 1.1 m），场地 56×56 m；curriculum 曲线显示 num_active 从 0 平滑升至
119.8/120，且该过程中训练成功率维持在 0.97 左右。

## 6. 统一评估协议与定量对比

协议：deterministic policy、seed 42、每格 64 envs × 4 episodes（256 episodes），
RandomDense 侧恒定 120 障碍、baseline 侧烘焙模板。

| 策略 \ 测试分布 | 固定烘焙地图 | 随机 120 障碍布局 |
|---|---|---|
| Baseline（固定图训练） | 0.992 / 跌倒 0 / 超时 0.8% | 0.988 / 跌倒 0 / 超时 1.2% |
| RandomDense（随机训练） | 0.992 / 跌倒 0 / 超时 0.8% | 0.973 / 跌倒 1.2% / 超时 1.6% |

（数值为成功率；末端目标误差四格均在 0.42–0.50 m。原始数据见 `outputs/eval/*.json`。）

**结论**：泛化假设未获支持——baseline 只见过 3 张固定图，但在从未见过的随机高密度布局上
仍有 0.988 成功率，说明池化高度扫描 + 相对目标观测下策略学到的是可迁移的局部避障反应
而非背图。RandomDense 在自身分布上略低（0.973），主要原因是其评估恒定最密档 120 障碍，
任务难度高于 baseline 的烘焙模板密度；两组差异（~1.5%）在 256 episodes 的统计误差边缘。
**代价**：RandomDense 全难度下约 14 s/iter，为 baseline（1.8 s/iter）的 8 倍，成功率无
显著提升——固定模板在本任务设定下已足够。

## 7. 定性结果

俯视图演示（绿圆 = 目标区，粉箭头 = 朝向指令，红点区 = 障碍软约束）：

- `figures/baseline_6s.png`：baseline 策略在固定模板中绕行圆柱与方墩；
- `figures/random_dense_6s.png`：RandomDense 策略穿越 120 障碍随机布局；
- 完整录像：`random_dense_on_random.mp4`、`baseline_on_fixed.mp4`（各 400 步）。

两组策略均能稳定行走、绕障并进入 0.5 m 成功半径，与定量结果一致。

## 8. 失败模式、局限与改进方向

1. **失败模式**：RandomDense 评估中的失败以超时为主（1.6%），跌倒（bad_orientation，1.2%）
   多发生在密集障碍间急转时——273 维 2×2 池化扫描在障碍极密处损失细节，可能低估窄间隙
   可通过性，策略倾向保守绕行直至超时。
2. **评估难度不对等**：RandomDense 评估恒定 120 障碍，baseline 烘焙模板密度较低，
   严格对比应统一障碍密度（可作为后续工作）。
3. **单 seed 局限**：全部实验仅 seed 42，0.988 vs 0.973 的差异不能排除随机波动，
   多 seed 重复是更严谨的做法。
4. **改进方向**：保留原始 1092 维扫描或用更小的池化核缓解感知损失；按成功率自适应
   调整障碍密度评估协议；尝试更长距离/多目标（方向 D）与动态障碍（方向 C）进一步
   push the limits。

## 附：复现要点

- 训练：`./unitree_rl_lab.sh -t --task <TaskID> --num_envs 512 --seed 42 --max_iterations 10000`
- 评估：`python scripts/rsl_rl/evaluate.py --headless --task <TaskID> --checkpoint <model_9999.pt> --num_envs 64 --episodes_per_env 4`
- 任务 ID：`Unitree-G1-29dof-Navigation-HRL-Baseline` / `Unitree-G1-29dof-Navigation-HRL-RandomDense`
- 最佳 checkpoint 路径与代码改动详见 `code/README_HW5.md`
