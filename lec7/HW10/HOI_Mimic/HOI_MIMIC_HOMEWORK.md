# 感知运控作业：HOI Terrain RayCaster

## 1. 作业目标

本作业要求为 G1-29dof HOI terrain mimic control 接入 RayCaster 地形高度感知，并通过训练和可视化验证感知链路。

本作业只使用以下感知任务：

```text
Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast
```

## 2. 感知运控链路

```text
*.terrain.json / mjcf_boxes
        ↓
HOI terrain mesh → torso_link RayCaster
        ↓
mdp.height_scan
        ↓
policy / critic observations → PPO policy
```

`mdp.height_scan` 使用 Isaac Lab locomotion 任务中的相对高度定义：

```text
height = sensor_pos_w.z - ray_hit_w.z - offset
```

默认扫描网格使用 `size=[1.6, 1.6]`、`resolution=0.1`，即单帧`17 × 17 = 289` 个采样点。`history_length=8` 由 observation manager 负责，不要在传感器或 observation term 中手工堆叠历史帧。

## 3. 作业范围

只需要完成三组核心 TODO：

1. 读取并校验 terrain metadata。
2. 配置 RayCaster height scanner。
3. 将 height scanner 接入 policy 和 critic。

相关文件：

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/
├── mdp/
│   └── hoi_height_scan.py
└── robots/g1_29dof/
    └── hoi_mimic_terrain_perceptive_raycast/
        └── tracking_env_cfg.py
```

项目已经提供 RayCaster mesh 构建、`mdp.height_scan`、任务注册、PPO 配置、训练和播放脚本，以及 height-scan 调试工具。

本作业不要求修改 `play.py`、helpers、RayCaster 实现、任务注册、PPO runner或数据转换脚本。确有必要修改范围外文件时，须在提交说明中解释原因。

## 4. 核心 TODO

### TODO 1：读取 terrain metadata

文件：

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/mdp/hoi_height_scan.py
```

函数：

```python
_normalize_box_record(entry, idx)
_load_boxes(metadata_file, device)
```

要求：

- 从 `*.terrain.json` 读取非空的 `mjcf_boxes`。
- 每个 box 包含：
  - `pos`：terrain body 局部坐标系中的位置，长度为 3；
  - `quat`：terrain body 局部坐标系中的姿态，顺序为 `wxyz`，长度为 4；
  - `half_size`：box 半边长，长度为 3。
- 如果只有 `full_size`，应转换为 `half_size`。
- 对非法数据给出包含 box 下标的明确错误。
- 返回 `torch.float32` tensor：
  - `box_pos_local`: `[num_boxes, 3]`
  - `box_quat_local`: `[num_boxes, 4]`
  - `box_half`: `[num_boxes, 3]`
- 返回 tensor 必须位于调用方指定的 device。
- 不得在每个仿真 step 中读取 JSON。

说明：该文件中其余 analytical height-scan TODO 不属于本作业，不需要实现。

### TODO 2：配置 RayCaster height scanner

文件：

```text
source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/
└── hoi_mimic_terrain_perceptive_raycast/tracking_env_cfg.py
```

在 `RobotSceneCfg` 中配置名为 `height_scanner` 的`HoiMergedTerrainRayCasterCfg`。核心要求：

- 挂载到 `{ENV_REGEX_NS}/Robot/torso_link`。
- 使用 `ray_alignment="yaw"`，扫描网格只随机器人 yaw 转动。
- 使用 `GridPatternCfg(resolution=0.1, size=[1.6, 1.6])`。
- `terrain_prim_path` 指向每个环境中的 `HOI_Terrain`。
- `metadata_file` 使用 blind task 已定义的 `TERRAIN_META_FILE`。
- 使用 `mjcf_boxes` 构建感知 mesh。
- 包含 ground plane，避免 box 之外的射线全部 miss。
- 默认地形位姿固定，设置 `rebake_on_reset=False`。

### TODO 3：接入 policy 和 critic observation

在同一 `tracking_env_cfg.py` 中，为 `PolicyCfg` 和 `PrivilegedCfg` 分别添加名为 `height_scanner` 的 `ObsTerm`。

两者共同要求：

- `func=mdp.height_scan`
- `sensor_cfg=SceneEntityCfg("height_scanner")`
- `offset=0.5`
- `clip=(-1.0, 5.0)`
- `history_length=blind_cfg.PROPRIO_HISTORY_LENGTH`

policy observation 额外加入：

```python
Unoise(n_min=-0.02, n_max=0.02)
```

critic observation 不添加该噪声。

## 5. 运行与验收

### 5.1 准备环境和数据

```bash
cd <project_path>
conda activate env_isaaclab
source set_project_root.sh
```

完整的数据准备流程见 [HOI_MIMIC_RUN_GUIDE.md](HOI_MIMIC_RUN_GUIDE.md)。运行前检查：

```bash
test -f "${HOI_MIMIC_TERRAIN_MOTION_FILE}" && echo OK || echo MISSING
test -f "${HOI_MIMIC_TERRAIN_META_FILE}" && echo OK || echo MISSING
test -f "${HOI_MIMIC_TERRAIN_URDF}" && echo OK || echo MISSING
```
### 5.2 Smoke train

先用少量环境检查任务创建、observation shape 和基本 step：

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
  --num_envs 4 \
  --max_iterations 10 \
  --headless \
  --logger tensorboard
```

验收要求：完成 10 次 iteration，且不出现路径、shape、`NotImplementedError`
或 `NaN/Inf` 错误。

### 5.3 RayCaster 可视化

使用一个环境打开 GUI：

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
  --num_envs 1 \
  --height-scan-vis \
  --height-scan-print
```

验收要求：扫描网格随机器人位置和 yaw 移动；命中点位于 ground plane 或 box表面附近；打印的高度统计为有限值。

### 5.4 训练与播放

以下命令是参考配置。教师可根据课程算力调整环境数和迭代数：

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
  --num_envs 1024 \
  --max_iterations 50000 \
  --headless \
  --logger wandb \
  --log_project_name unitree_hoi_mimic_terrain_perceptive_raycast
```

播放训练得到的 checkpoint：

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast \
  --checkpoint <checkpoint_path> \
  --num_envs 1 \
  --hoi-play-no-curriculum \
  --hoi-play-no-dr \
  --height-scan-vis \
  --height-scan-print
```

## 6. 评分

| 项目 | 分值 | 验收重点 |
|------|-----:|----------|
| TODO 1：metadata loader | 20 | 数据读取、校验、shape、dtype 和 device 正确 |
| TODO 2：RayCaster 配置 | 20 | 挂载、扫描网格、地形 mesh 和 reset 策略正确 |
| TODO 3：observation 接入 | 20 | policy/critic 配置、history、noise 和 clipping 正确 |
| Smoke train 与数值检查 | 10 | 能连续运行且无 `NaN/Inf` |
| RayCaster 可视化 | 10 | 扫描点和高度统计合理 |
| 训练与播放 | 15 | 生成 checkpoint 并成功播放 |
| 代码质量与改动范围 | 5 | 实现清晰，无无关改动 |

## 7. 提交内容

- 修改后的 `hoi_height_scan.py` 和 Raycast `tracking_env_cfg.py`。
- Smoke train 的终端输出。
- RayCaster 命中点截图或录屏。
- 训练日志、checkpoint 路径和播放结果。
- `git diff --stat`，用于确认没有修改作业范围之外的代码。
