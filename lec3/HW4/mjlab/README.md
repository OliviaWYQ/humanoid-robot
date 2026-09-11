# mjlab：实践 4 蹲姿行走

此目录是深蓝学院实践 4 的 mjlab 作业代码，任务为 Unitree G1 速度 + 骨盆高度跟踪。
TODO 位置、实现及验证记录见 [作业说明](docs/HOMEWORK_TODO.md)。

## macOS CPU 运行

```bash
uv sync --locked --extra cpu
uv run list-envs
uv run train Mjlab-VelocityHeight-Flat-Unitree-G1 \
  --gpu-ids None --env.scene.num-envs 2 \
  --agent.max-iterations 1 --agent.logger tensorboard
```

以上是训练链路检查，不会得到收敛的行走策略。Mac 使用 Warp CPU 后端；
文档中的 4096 并行环境面向 Linux + NVIDIA GPU 完整训练。

## 检查与 Linux GPU 训练

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_velocity_task.py tests/test_velocity_height_mdp.py tests/test_gpu_selection.py -q
uv run python scripts/check_velocity_height.py

# 在具有 NVIDIA GPU 的 Linux 主机上运行完整训练
uv sync --locked --extra cu128
uv run --extra cu128 train Mjlab-VelocityHeight-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 --agent.logger tensorboard
```

依赖已固定到验证过的 MuJoCo 3.8.1 / MuJoCo Warp 3.9.0.1 正式发行包，
避免原 nightly wheel 的 404。`uv run` 会按 `uv.lock` 同步依赖。

基于 [mujocolab/mjlab](https://github.com/mujocolab/mjlab)，使用 Apache-2.0 许可证。
