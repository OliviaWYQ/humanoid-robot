# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch evaluation of an RSL-RL checkpoint under a fixed evaluation protocol.

Runs a trained policy deterministically for a fixed number of episodes per
parallel environment and reports outcome fractions (success / fall / timeout),
episode length, and final goal error. Used for the Part 2 baseline-vs-extension
comparison; both policies are evaluated with the identical protocol here.

Example:
    python scripts/rsl_rl/evaluate.py --task Unitree-G1-29dof-Navigation-HRL-RandomDense \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_9999.pt --num_envs 64 --episodes_per_env 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Batch-evaluate an RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Evaluation task (defines the test distribution).")
parser.add_argument("--episodes_per_env", type=int, default=4, help="Completed episodes collected per environment.")
parser.add_argument("--seed", type=int, default=42, help="Evaluation seed (protocol fixed at 42 for the report).")
parser.add_argument("--output", type=str, default=None, help="Optional path to write a JSON summary.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import cli_args
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    torch.manual_seed(args_cli.seed)

    # Use the PLAY entry point: it disables curriculum and (for the random-layout
    # task) pins the full 120-obstacle layouts, so evaluation runs at the final
    # difficulty for every task.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    resume_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=100.0)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    term_mgr = raw_env.termination_manager
    outcome_keys = ["goal_reached", "base_height", "bad_orientation", "time_out"]
    for key in outcome_keys:
        if key not in term_mgr.active_terms:
            raise KeyError(f"Termination term '{key}' not found in task '{args_cli.task}'.")

    total_target = args_cli.num_envs * args_cli.episodes_per_env
    outcomes = {key: 0 for key in outcome_keys}
    episode_lengths = []
    final_errors = []
    completed = 0
    steps = 0
    max_steps = total_target * 60  # generous upper bound (episode_length_s / step_dt ~= 150)

    # Per-env step counters (episode_length_buf is zeroed by the reset that
    # happens inside env.step when an episode ends, so we count ourselves).
    env_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=raw_env.device)
    # Termination buffers are not cleared by the in-step reset and still hold
    # the ended episode's fired terms right after env.step returns; they are
    # overwritten by the next compute(), so we must read them immediately.
    # The goal-error metric, in contrast, already reflects the resampled goal,
    # so we snapshot it before each step (<= one high-level step of lag).
    error_before = None

    obs = env.get_observations()
    while simulation_app.is_running() and completed < total_target and steps < max_steps:
        error_metric = raw_env.command_manager.get_term("pose_command").metrics.get("error_pos_2d")
        if error_metric is not None:
            error_before = error_metric.clone()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        env_steps += 1
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            truncated = term_mgr.time_outs
            for idx in done_ids.tolist():
                if bool(truncated[idx]):
                    outcomes["time_out"] += 1
                else:
                    for key in ("goal_reached", "base_height", "bad_orientation"):
                        if bool(term_mgr.get_term(key)[idx]):
                            outcomes[key] += 1
                            break
                episode_lengths.append(float(env_steps[idx]) * raw_env.step_dt)
                if error_before is not None:
                    final_errors.append(float(error_before[idx]))
                env_steps[idx] = 0
            completed += len(done_ids)
        steps += 1

    summary = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "episodes_per_env": args_cli.episodes_per_env,
        "episodes_completed": completed,
        "success_rate": outcomes["goal_reached"] / max(completed, 1),
        "fall_rate": (outcomes["base_height"] + outcomes["bad_orientation"]) / max(completed, 1),
        "timeout_rate": outcomes["time_out"] / max(completed, 1),
        "mean_episode_length_s": sum(episode_lengths) / max(len(episode_lengths), 1),
        "mean_final_goal_error_m": sum(final_errors) / max(len(final_errors), 1),
        "outcome_counts": outcomes,
    }
    print("[EVAL SUMMARY]")
    print(json.dumps(summary, indent=2))

    if args_cli.output:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        with open(args_cli.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Summary written to: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
