# 实践 7：SMPL-X → G1 29DoF → AMP

## 文件

- `scripts/gmr_npz.py`：公共转换、CPU 正运动学、格式检查。
- `scripts/smplx_to_robot_npz.py`：单条转换，默认启动 GMR MuJoCo 窗口。
- `scripts/smplx_to_robot_dataset_npz.py`：批量转换，保留目录结构，支持 spawn 多进程；失败返回非零状态。
- `scripts/vis_robot_motion_npz.py`：窗口播放及离屏 MP4 录制。
- `scripts/verify_npz.py`：格式、关节范围、独立 FK、接触诊断及 HW8 实际加载测试。
- `output/`：新生成的动作与验证报告。

脚本同时安装在 `/Users/mac/Desktop/github/GMR/scripts/`。
可通过 `GMR_ROOT` 指定其他 GMR 路径。

## 本机复现

在 humanoid-robot 根目录执行：

```bash
source lec5/HW7/.venv/bin/activate
python lec5/HW7/submission/scripts/smplx_to_robot_dataset_npz.py \
  --src_folder lec5/HW7/submission/input \
  --tgt_folder lec5/HW7/submission/output \
  --body_models /Users/mac/Desktop/github/GMR/assets/body_models \
  --num_cpus 1 --override
python -m unittest discover -s lec5/HW7/submission/tests -v
python lec5/HW7/submission/scripts/verify_npz.py \
  lec5/HW7/submission/output \
  --amp_loader lec5/HW8/unitree_lab_amp-main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp/motion_dataset.py \
  --report lec5/HW7/submission/output/validation.json
```

Mac 的 MuJoCo 交互窗口需要用环境中的 `mjpython`：

```bash
bash lec5/HW7/submission/play_hw7.sh
```

提供的 `play_hw7.sh` 也会补充 uv Python 的动态库路径，避免 macOS `libpython3.10.dylib` 找不到的问题。

Linux 可以用 `python` 启动交互窗口。离屏视频命令：

```bash
python lec5/HW7/submission/scripts/vis_robot_motion_npz.py \
  --robot_motion_path lec5/HW7/submission/output/B3_-_walk1_stageii.npz \
  --headless --video lec5/HW7/submission/output/walk.mp4
```

## 实现说明

人体有 SMPL-X 骨架、手指和面部参数；G1 是固定的 29 个旋转关节，连杆长度、肩髋零位和关节范围不同，不能直接复制人体关节角。转换保留 GMR 的 `smplx_to_g1.json` 两阶段 IK 映射及高度/局部比例缩放。人体 pelvis 映射到 G1 pelvis，脚映射到脚踝，腕映射到腕部；配置中的旋转偏置完成零位对齐。IK 使用 DAQP 及 GMR 的关节位置范围约束。

SMPL-X 参数在时间轴上通过旋转 SLERP 和平移线性插值重采样；首尾时间戳与输入一致，实际帧率按 `(T-1)/duration` 保存。人体模型按 32 帧分块计算，关闭梯度，避免整条高分辨率人体网格同时占用内存。首帧先迭代收敛，再从第 0 帧记录，不丢首帧。

输出根位置取 IK 求得的机器人 qpos，首帧 x/y 归零；姿态从 wxyz 转 xyzw、归一化并保持相邻四元数同半球。29DoF 按 XML 名称严格检查，额外保存 Unicode `joint_names`。`local_body_pos` 由 GMR KinematicsModel 在零平移、单位根旋转下计算，CPU 可运行。所有输出通过 `allow_pickle=False` 验证，没有 object/None。

验证器用 MuJoCo 世界位置独立计算 `R_root^-1 (p_link - p_root)`，逐帧核对局部位置。脚底高度取脚上四个碰撞球的最低点；近地脚速度是高度小于 4 cm 的连续帧的水平速度，属于滑脚诊断量，不能单凭该量判断真实接触。自碰撞指标基于模型启用的碰撞几何，不覆盖所有可视网格。视频用于进一步检查方向、节奏、穿模与脚部接触。

IK 生成的是运动学参考动作。实际动力学稳定性和 AMP 策略训练需要实践 8 的仿真训练验证。

## 迁移到其他环境

创建 Python 3.10 环境，安装 `requirements-lock.txt`（记录本次验证的完整版本）；将 `scripts/*.py` 复制到 GMR 的 `scripts/`。在 GMR 根目录应用 `patches/gmr_compatibility.patch`。该补丁固定 NumPy/SciPy，补充 torch/DAQP，并修复新版 mink 的 IK 参数传递：所有 `solve_ik` 调用显式传入 `limits=self.ik_limits`。仅复制导出脚本而不修复此调用，会使新增约束失效。

本次 PyTorch 2.11.0 从本机现有 Python 3.10 环境复制，其余依赖在独立 `.venv` 中安装；原有环境未修改。

## 动作质量处理

对两只手与躯干启用 Mink 距离约束（目标间距 8 mm、检测距离 5 cm）。在每个输出帧开始时建立关节可达区间，腕部帧间速度上限为 6 rad/s，其余关节为 20 rad/s。区间在整帧的多次 IK 迭代内保持固定，因此约束的是输出帧之间的总变化。这些上限是本作业的运动学连续性设置，不代表电机经过验证的硬件速度额定值。没有对结果直接裁剪或对四元数做欧拉角滤波。

## 实践 8

四条输出已复制到 `lec5/HW8/unitree_lab_amp-main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp/data/hw7/`。
`motion_cfg.py` 新增 `G1HW7MotionCfg` 和 `MOTION_CONFIGS["hw7"]`，四条动作采样权重均为 1。
可通过 `make_motion_source("hw7")` 及 `make_motion_dataset_cfg(...)` 使用；调用加载器前应把 `joint_names` 设置为实际机器人关节名称。现有 walk/run/mixed 配置保持原有选择。

运行 `bash lec5/HW7/submission/run_hw7.sh --override` 可重新生成、测试和录制；图形录制需要可用的桌面 OpenGL 环境。

## 腕部修正与新增转弯动作

当前数据包括行走 B3、左转 B9、跑步 C3、走转跑 C5。B9 文件名标注左转 90°，源动作实际 pelvis 朝向变化约 102.07°，机器人约 102.75°；保留源动作实际转向。

`add_wrist_constraints` 在初始化求解前生效，两阶段 IK 均加入零位腕部姿态代价 5；腕部朝向跟踪代价降低为 0.5，保持原来的腕部位置跟踪。另加独立的腕部角度约束：roll ±45°、pitch ±30°、yaw ±20°。原有每帧速度及手部避碰约束继续生效。这些是针对 locomotion 的设置，会弱化手势细节，不建议直接用于精细操作动作。

验证器除检查速度之外，还检查腕部角度和贴近限位的帧比例（距限位 1°内达到 25% 即失败），避免静止卡在极限姿态也通过验证。每个腕关节的实际范围、贴限位比例和最长连续帧数见 `output/validation.json`。
