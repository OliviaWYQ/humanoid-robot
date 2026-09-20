# 实践 6：教师–学生蒸馏

作业理论、Linux 训练及评估命令见 [ASSIGNMENT.md](ASSIGNMENT.md)。

## 已完成的代码

- `distillation_utils.py`：线性退火、MSE/Huber、action MAE/RMSE、数值稳定的 forward KL、Gaussian 参数 RMSE。
- `action_matching.py`：教师无梯度推理、学生 action 回归、日志指标和退火。
- `kl_matching.py`：KL(teacher || student)、学生 mean/std 梯度、日志指标和退火。
- 保留 `distillation_ppo.py` 的共享 PPO 循环。
- 原测试引用未随学生包提供的 `instructor_solutions`；已改为测试实际提交代码，另加 PyTorch 分布 KL 对照、边界和梯度检查。

## 为什么 Mac 上找不到 mjlab

本机的 HW4 环境位于 `../../../../lec3/HW4/mjlab/.venv`，不在 Conda 环境列表中。
默认 `python3` 是 Homebrew Python 3.14，而本作业要求 Python 3.10–3.13。
HW4 安装的是 mjlab 1.4.0，HW6 固定为 1.2.0；直接用 HW4 环境会报
`ImportError: cannot import name 'DelayedActuatorCfg'`。应使用独立环境。
此外 mjlab 1.2.0 的依赖声明漏了地形模块使用的 SciPy；Mac 依赖文件已显式补上。

## macOS 独立环境

在本目录执行。`.venv-macos` 与 HW4 的环境互不影响：

```bash
uv venv .venv-macos --python 3.13
uv pip install --python .venv-macos/bin/python -r requirements-macos.txt
uv pip install --python .venv-macos/bin/python --no-deps -e .
source .venv-macos/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m mjlab.scripts.list_envs
```

日常使用只需 `source .venv-macos/bin/activate`。此环境请直接用 `python`，
`uv run` 默认会创建/同步另一个 `.venv`，不会自动使用 `.venv-macos`。

参考 motion 的可视化不需要并行物理训练：

```bash
python -m humanoid_hw6.scripts.data.visualize_motion_curate_viser \
  --motion assets/motions/g1_accad_walk --host 127.0.0.1
```

Mac 的 Warp 后端只有 CPU，没有 Metal/MPS 加速。可以做算法开发、单元测试、
参考动作浏览；完整训练按作业要求使用 Linux + NVIDIA GPU。
仅安装 Linux 虚拟机或 Docker 不会让 Apple GPU 获得 CUDA 能力。

## 本机实际验证（2026-09-15）

- Python 3.13.12 / mjlab 1.2.0 / RSL-RL 5.0.1；`uv pip check` 通过。
- 全部 **24 个 pytest 测试通过**；修改文件的 Ruff 检查通过。
- 三个 HW6 task ID 注册成功；25 段 motion 共 5548 帧可加载。
- Action Matching 和 KL Matching 都完成 **CPU、2 个环境、1 次 PPO 更新**，
  成功加载课程教师权重，输出蒸馏指标、TensorBoard 日志、checkpoint 和 ONNX。
- 这是链路验证，不能作为收敛效果或两种算法优劣的实验结论。

复现最小训练检查（把 task 中的 `Action-Matching` 换为 `KL-Matching` 即可检查另一种）：

```bash
WARP_CACHE_PATH=/tmp/hw6-warp MPLCONFIGDIR=/tmp/hw6-mpl WANDB_MODE=disabled \
python -m mjlab.scripts.train Mjlab-Humanoid-HW6-Student-Action-Matching-G1 \
  --gpu-ids None --env.scene.num-envs 2 \
  --agent.max-iterations 1 --agent.logger tensorboard \
  --env.commands.motion.motion-file src/humanoid_hw6/config/g1/motion_data_cfg_g1_accad_walk.yaml \
  --agent.teacher-checkpoint-file checkpoints/g1_hw6_teacher/model_latest.pt \
  --agent.run-name macos_cpu_smoke
```

`--gpu-ids None` 显式使用 CPU。缓存目录变量用于绕开工具沙箱对用户缓存目录的写限制，
普通终端通常可以省略。第一次运行需要编译 Warp CPU kernel。

本次成功运行的产物：

- `logs/rsl_rl/g1_hw6_student_action_matching/2026-09-15_19-55-23_macos_cpu_smoke/`
- `logs/rsl_rl/g1_hw6_student_kl_matching/2026-09-15_19-56-07_macos_cpu_smoke/`

## Linux + NVIDIA GPU

在 Linux 主机上创建独立环境，使用作业原始锁文件：

```bash
uv sync --locked --dev
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q
```

训练命令和完整评估协议见 `ASSIGNMENT.md` 第 7 节。建议先将环境数降至 2、
迭代数降至 1，检查教师加载和两种算法链路，再执行正式训练。
当前代码验证不等于完成了两组收敛训练；实验报告需要实际训练日志和回放结果。

## 依据

- 作业 PDF 第 4 页：Linux、NVIDIA GPU、Python 3.10–3.13。
- [mjlab 1.2.0](https://github.com/mujocolab/mjlab/tree/v1.2.0)：安装包说明 macOS 支持评估，训练要求 NVIDIA GPU。
- [NVIDIA Warp 安装说明](https://nvidia.github.io/warp/latest/user_guide/installation.html)：Apple Silicon 上支持 CPU，不支持 Metal 加速。
