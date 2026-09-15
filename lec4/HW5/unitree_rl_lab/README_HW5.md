# HW5：TODO 实现与验证

## 目录怎么用

按作业 PDF 第 3 节，直接使用本目录这个完整学生包，不要与原版
`/Users/mac/Desktop/github/projects/unitree_rl_lab` 逐文件合并。
学生包包含配套 navigation、low-level locomotion、训练入口和预训练权重；
原版项目没有这些导航任务，配置和入口也有差异。
Linux 复用原有 `env_isaaclab`、Isaac Lab 和 Unitree 模型资产即可。

本次完成 Part 1 的 7 个 TODO（3 个源文件），保留所有 START/END 标记。
未修改低层训练任务、权重、训练/play 入口；尚未进行训练或 Part 2 开放实验。

## 本地 CPU smoke

在本目录运行，Python 需要安装 PyTorch：

```bash
python scripts/smoke_hw5_cpu.py
```

本次已在 macOS / Python 3.12 / PyTorch 2.14.0 CPU 上运行通过。
临时解释器为 `/private/tmp/hw5-smoke-venv/bin/python`（临时目录可能被系统清理）。

无需安装 Isaac Sim/Isaac Lab。脚本从源文件提取实际函数和配置类，
以简化接口替代模拟器依赖，检查：

- 全项目 Python 语法和 TODO 占位异常清理；
- 二维扫描 26×42 → 13×21 → 273 维池化（逐窗口核对数值）；
- 高层 actor 376 维配置及缩放；
- deepcopy 后关闭低层噪声，保留 5 帧历史且不污染训练配置；
- 真实 `policy.pt` 在 CPU 上接受 480 维输入，输出 29 维有限动作；
- eval、冻结参数、inference_mode，速度逐维裁剪及输入不被改写；
- 40 个物理步执行 10 次低层推理、40 次动作应用，以及部分/全部 reset。

输入历史在此测试中是合成数据。该测试不验证真实 ObservationManager 的历史顺序、
传感器、USD 资产、CUDA、物理稳定性或 PPO update，不能替代下面的 Linux smoke。

## Linux 接续

把整个作业目录复制到 Linux 的独立目录，例如 `~/projects/unitree_rl_lab_hw5`。
以下路径需要替换为设备上的实际值：

```bash
conda activate env_isaaclab
cd ~/projects/unitree_rl_lab_hw5
export ISAACLAB_PATH=~/projects/IsaacLab
./unitree_rl_lab.sh -i
./unitree_rl_lab.sh -l
```

确认列表出现 LowLevel、Navigation-HRL-Baseline、Navigation-HRL-Extension 三个任务。
同一个环境中两个同名 `unitree_rl_lab` 包的 editable 安装会切换导入来源；
安装后检查下面输出应指向 HW5 目录：

```bash
python -m pip show unitree_rl_lab
```

**资产路径需要先核对**：学生包 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
中的 `UNITREE_MODEL_DIR` 目前写死为课程机器路径
`/home/sustech/桌面/unitree_locomotion_cb/unitree_model`。
在 Linux 上将该常量设为已有 Unitree 模型根目录，确认其下有
`G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd`。
这里没有猜测或修改你的 Linux 路径。

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); import isaaclab, rsl_rl; print('imports: OK')"
test -s pretrained/g1_29dof_lowlevel/policy.pt
./unitree_rl_lab.sh -t \
  --task Unitree-G1-29dof-Navigation-HRL-Baseline \
  --num_envs 16 \
  --max_iterations 1
```

脚本的 `-t` 已添加 `--headless`。成功标准：完成 1 次 PPO iteration，
没有 TODO、shape/device/manager 或显存错误。尚未在 Linux 执行此验收。

默认低层权重相对本项目定位；如需显式指定，可设置
`UNITREE_G1_LOW_LEVEL_POLICY_PATH`。无需重新下载权重或训练低层策略。

## 接口速查

- 物理 200 Hz；低层 50 Hz（每 4 个物理步）；高层 5 Hz（每 40 个物理步）。
- 低层每帧：3 + 3 + 3 + 29 + 29 + 29 = 96，5 帧共 480。
- 高层 actor：3 + 3 + 3 + 4 + 3 + 29 + 29 + 29 + 273 = 376。
- critic 另有 base height、command distance，共 378。
