# HoST 身体扭曲诊断

日期：2026-09-05。未修改部署脚本、配置、策略或原录像。

## 结论

在无窗口、无渲染、无实时 sleep 的仿真中，调用现有部署脚本的四个函数复现了站起后的扭曲。现有四个 TODO 符合 starter 的实现提示；没有发现明确的状态切片、重力计算、历史拼接或动作增量公式错误。

目前确认的是：策略与当前模型组成的闭环收敛到了不自然、多个动作输出饱和的站姿。预训练 checkpoint / 导出配套信息缺失，因此还不能将根因唯一归结为训练质量、导出问题或训练与部署模型/接触动力学差异。不能据此断言关节顺序错乱，也不建议直接对腰部动作加偏置掩盖问题。

## 复现条件与结果

本机 Python 环境：MuJoCo 3.12.0，PyTorch 2.14.0；CPU 推理。
物理步长 0.001 秒，策略周期 0.020 秒，初始状态采用 XML，策略历史与首段 PD 与主程序一致。
基线运行 60 秒仿真时间。

| 指标 | 60 秒时 |
|---|---:|
| 基座高度 | 0.741 m |
| 基座相对竖直方向倾角 | 2.33° |
| 腰部 yaw | +24.49° |
| 左 / 右髋 yaw | -33.5° / -33.2° |
| 左 / 右踝 roll | -15.8° / +16.5° |
| 右膝 | -6.2° |

最后 10 秒，左踝 roll、右膝、右踝 roll、左肩 roll 的策略输出绝对值始终大于 0.95（网络末层为 Tanh）。双踝的 XML 范围约为 ±15°，右膝下限约为 -5°。小幅越过范围不等于没有限位；当前仿真约束允许一定偏差。动作持续饱和且姿态近乎静止，说明存在持续受力，不能把低速度误解为自然、放松的站姿。

## 排查与对照

- 本地模型 23 个关节的 qpos 顺序与 23 个 actuator 的顺序一致。
- 官方公开 G1 URDF 的非固定关节在文件中的排列与本地 XML 相同；这不等于已经验证该 checkpoint 的训练运行时 joint_names。
- 重力函数与随机 100 个单位四元数对应的 R.T @ [0, 0, -1] 对照，最大误差 2.98e-8。
- 观测布局、缩放、旧到新的历史更新，与 HoST 官方公开 host_ground.py 一致。
- 作业 PDF 第 4 页公式把最新帧写在前面，但第 13 页 TODO、第 17 页评分要求最新帧在末尾。当前代码采用后者；反转历史的 15 秒对照不能稳定站起。

以下均为临时测试，未写入项目代码。除基线外，每项运行 15 秒仿真：

| 对照 | 腰部最终 yaw | 结论 |
|---|---:|---|
| 原配置（15 秒） | 23.83° | 可稳定复现 |
| 将 default_angles 写入初始 qpos | 23.62° | 未解决 |
| 肩部软件力矩限制改为 XML 的 25 Nm | 23.83° | 与基线相同 |
| 每个物理步按当前 q 重算增量目标 | 23.78° | 起身过程改变，最终扭转仍在 |
| 摩擦系数改为 0.7 | 23.02° | 未解决 |

## 框架中另外发现的问题

1. default_angles 只用于初始 PD 目标，没有写入 mj_data.qpos；所以注释中“初始关节角”的说法不准确。但对照显示它不是这次持续扭转的主因。
2. YAML 左肩 pitch / roll 力矩限制为 50 Nm，而 XML 对这两个关节限制为 25 Nm；XML 仍执行限制，统一数值没有改变基线。
3. 官方 host_ground.py 每个物理步用当前 q 加动作增量，starter 则每个策略步更新目标，并在 20 个物理步内保持目标。两者有控制语义差异，但本次对照未解决最终姿态。
4. simulation_duration 比较的是墙钟时间；录像与 viewer 开销会让实际仿真时间短于配置的 60 秒。现有 simulation.mp4 为 5 秒。这影响观察时长，不是持续扭转的解释。

## 建议的下一步

保留当前四个 TODO。优先取得该 pretrained_humanoid_standup.pt 对应的训练配置、运行时关节名顺序、导出脚本，以及同一个 checkpoint 在原训练环境或课程参考部署中的录像。以同一 checkpoint 对齐输入、输出与动力学，才能区分策略本身的姿态偏差与 sim-to-sim 偏差。

增量公式 q_target = q_current + scale * action 并不会主动将姿态拉回 default_angles；自然的站立姿态需要策略学会输出相应的纠偏动作。把基准直接换成 default_angles 会改变动作空间，不能作为此策略的直接修复。

## 参考

- HoST 官方训练观测、历史与控制：https://github.com/InternRobotics/HoST/blob/main/legged_gym/legged_gym/envs/base/host_ground.py
- HoST 官方 G1 配置：https://github.com/InternRobotics/HoST/blob/main/legged_gym/legged_gym/envs/g1/g1_config_ground.py
- MuJoCo 自由关节角速度的局部坐标系：https://mujoco.readthedocs.io/en/stable/overview.html#floating-objects

## 本次分析的文件 SHA-256

- `deploy_mujoco_host_student.py`: `ae42c20b5e71400e69abffdcc3f8241668fe2f663af4b38439f378bc00c29c6d`
- `configs/g1.yaml`: `a2919dbb50adaa671be092aa4b2f9733efc1cf940df83ea55b5aff187bce0fe4`
- `robots/g1/g1_23dof.xml`: `d3fed831dd2deb06ca07a96a2bafb23dd482970ea9fc4f40776900384057afc0`
- `policies/pretrained_humanoid_standup.pt`: `6021fed199d0d1990471248ff92e3408296d5105db6dc3c2fbe6bffc2118165b`
