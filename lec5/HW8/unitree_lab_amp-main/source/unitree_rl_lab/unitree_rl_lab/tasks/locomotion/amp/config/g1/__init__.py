"""Gym registrations for the readable G1 AMP configuration tree."""

from __future__ import annotations

import gymnasium as gym

from . import agents


ENV_ENTRY_POINT = "unitree_rl_lab.tasks.locomotion.amp.amp_env:AMPManagerBasedRLEnv"
ENV_CFG_MODULE = f"{__name__}.amp_flat_env_cfg"
AGENT_CFG_MODULE = f"{agents.__name__}.rsl_rl_ppo_cfg"


def register_g1_amp_task(task_id: str, train_cfg: str, play_cfg: str, runner_cfg: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point=ENV_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{ENV_CFG_MODULE}:{train_cfg}",
            "play_env_cfg_entry_point": f"{ENV_CFG_MODULE}:{play_cfg}",
            "rsl_rl_cfg_entry_point": f"{AGENT_CFG_MODULE}:{runner_cfg}",
            "amp_rl_cfg_entry_point": f"{AGENT_CFG_MODULE}:{runner_cfg}",
        },
    )


G1_AMP_TASKS = (
    ("Unitree-G1-29dof-AMP", "G1AMPMixedEnvCfg", "G1AMPMixedPlayEnvCfg", "G1AMPMixedRunnerCfg"),
    ("Unitree-G1-29dof-AMP-Walk", "G1AMPWalkEnvCfg", "G1AMPWalkPlayEnvCfg", "G1AMPWalkRunnerCfg"),
    ("Unitree-G1-29dof-AMP-Run", "G1AMPRunEnvCfg", "G1AMPRunPlayEnvCfg", "G1AMPRunRunnerCfg"),
    (
        "Unitree-G1-29dof-AMP-OmniRun",
        "G1AMPOmniRunEnvCfg",
        "G1AMPOmniRunPlayEnvCfg",
        "G1AMPOmniRunRunnerCfg",
    ),
    (
        "Unitree-G1-29dof-AMP-WalkToRun",
        "G1AMPWalkToRunEnvCfg",
        "G1AMPWalkToRunPlayEnvCfg",
        "G1AMPWalkToRunRunnerCfg",
    ),
    # TODO: 注册 WalkToRun-FullPlay 评估任务。
    # 提示：需要在 amp_flat_env_cfg.py 中先补全 G1AMPWalkToRunFullPlayEnvCfg。
    ("Unitree-G1-29dof-AMP-Dance", "G1AMPDanceEnvCfg", "G1AMPDancePlayEnvCfg", "G1AMPDanceRunnerCfg"),
    ("Unitree-G1-29dof-AMP-Play", "G1AMPPlayEnvCfg", "G1AMPPlayEnvCfg", "G1AMPPlayRunnerCfg"),
    ("AMP-Flat-G1-walk-v0", "G1AMPWalkEnvCfg", "G1AMPWalkPlayEnvCfg", "G1AMPWalkRunnerCfg"),
    ("AMP-Flat-G1-run-v0", "G1AMPRunEnvCfg", "G1AMPRunPlayEnvCfg", "G1AMPRunRunnerCfg"),
    ("AMP-Flat-G1-dance-v0", "G1AMPDanceEnvCfg", "G1AMPDancePlayEnvCfg", "G1AMPDanceRunnerCfg"),
    ("AMP-Flat-G1-Play-v0", "G1AMPPlayEnvCfg", "G1AMPPlayEnvCfg", "G1AMPPlayRunnerCfg"),
)

for task in G1_AMP_TASKS:
    register_g1_amp_task(*task)
