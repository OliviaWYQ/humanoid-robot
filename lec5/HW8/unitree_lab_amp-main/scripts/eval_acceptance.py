# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HW8 acceptance evaluation for the AMP walk-to-run policy.

Drives the FullPlay env with FIXED velocity commands (instead of the random
command generator) so each acceptance scenario is tested explicitly, records
one video covering scenarios 1-6, and prints a per-scenario metrics table:

  1. low-speed forward    2. medium-speed forward   3. high-speed forward
  4. decelerate (run->stop)   5. left turn   6. right turn   7. long run

Metrics per scenario: mean |lin-vel error|, mean |ang-vel error|, falls,
mean foot-slide (same mdp.feet_slide used as a training reward), mean |pitch
proxy| (projected gravity x), cadence + stride at contact events, and max
joint position deviation / velocity (divergence check).

Usage (same env as train/play):
  python scripts/eval_acceptance.py --headless \
    --task Unitree-G1-29dof-AMP-WalkToRun-FullPlay \
    --checkpoint logs/rsl_rl_amp/unitree_g1_29dof_amp_walk_to_run/<run>/model_29999.pt \
    --num_envs 1 --video
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Acceptance eval for the AMP walk-to-run policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during eval.")
parser.add_argument(
    "--video_length", type=int, default=9000, help="Steps recorded (default covers scenarios 1-6)."
)
parser.add_argument(
    "--only", type=str, default=None, help="Run only this scenario by name (e.g. 1_low_speed)."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Make the checkout runnable without installing the local Isaac Lab extension.
import pathlib  # noqa: E402

unitree_extension_path = pathlib.Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"
sys.path.insert(0, str(unitree_extension_path))

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

from rsl_rl_amp.env import AMPIsaacLabVecEnvWrapper  # noqa: E402
from rsl_rl_amp.runners import AMPRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.locomotion import mdp as unitree_mdp  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

# Acceptance scenarios: name -> list of (vx, wz, steps) phases.
# dt = 0.02 s, so 1500 steps = 30 s.
SCENARIOS = [
    ("1_low_speed", [(0.5, 0.0, 1500)]),
    ("2_medium_speed", [(1.2, 0.0, 1500)]),
    ("3_high_speed", [(3.0, 0.0, 1500)]),
    ("4_decelerate", [(3.0, 0.0, 500), (0.0, 0.0, 1000)]),
    ("5_left_turn", [(0.8, 0.8, 1500)]),
    ("6_right_turn", [(0.8, -0.8, 1500)]),
    ("7_long_run", [(1.2, 0.0, 7500)]),
]


def main():
    """Run the acceptance scenarios and report metrics."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.checkpoint is None:
        raise SystemExit("--checkpoint is required")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording (covers scenarios 1-6; the long run is not recorded)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "acceptance", args_cli.only or "all"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video of scenarios 1-6.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = AMPIsaacLabVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # handles into the simulation state
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene["contact_forces"]
    ankle_ids = contact.find_bodies(".*ankle_roll.*")[0]

    # take over the command generator: fix the command buffer ourselves
    cmd_buf = env.unwrapped.command_manager.get_command("base_velocity")
    term = env.unwrapped.command_manager.get_term("base_velocity")
    # In this Isaac Lab version the resample countdown lives on term.time_left
    # (tensor per env) and the range on term.cfg; disable both paths.
    term.cfg.resampling_time_range = (1e9, 1e9)
    if hasattr(term, "time_left"):
        term.time_left[:] = 1e18

    def set_command(vx: float, wz: float) -> None:
        cmd_buf[:] = torch.tensor([vx, 0.0, wz], device=cmd_buf.device, dtype=cmd_buf.dtype)
        if hasattr(term, "time_left"):
            term.time_left[:] = 1e18

    def foot_slide() -> float:
        try:
            value = unitree_mdp.feet_slide(
                env.unwrapped,
                asset_cfg=SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
                sensor_cfg=SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            )
            return float(value.mean().item())
        except Exception as error:  # noqa: BLE001
            print(f"[WARN] feet_slide unavailable: {error}")
            return float("nan")

    obs = env.get_observations()
    results = {}
    total_steps = 0

    scenarios = [s for s in SCENARIOS if args_cli.only is None or s[0] == args_cli.only]
    for name, phases in scenarios:
        acc = {
            "steps": 0,
            "lin_err": 0.0,
            "ang_err": 0.0,
            "falls": 0,
            "slide": 0.0,
            "pitch": 0.0,
            "contact_ons": 0,
            "speed_sum": 0.0,
            "max_joint_dev": 0.0,
            "max_joint_vel": 0.0,
            "nan_obs": False,
        }
        prev_contact = False
        print(f"\n=== scenario {name} ===")
        for vx, wz, n in phases:
            set_command(vx, wz)
            print(f"  command vx={vx:+.2f} wz={wz:+.2f} for {n} steps ({n * 0.02:.0f} s)")
            for _ in range(n):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, terminated, _ = env.step(actions)
                acc["steps"] += 1
                total_steps += 1
                if total_steps == 1:  # one-time sanity check that the override holds
                    print(
                        f"  [check] cmd={cmd_buf.tolist()} "
                        f"time_left={term.time_left.tolist() if hasattr(term, 'time_left') else 'n/a'}"
                    )
                vx_act = float(robot.data.root_lin_vel_b[:, 0].item())
                wz_act = float(robot.data.root_ang_vel_b[:, 2].item())
                acc["lin_err"] += abs(vx - vx_act)
                acc["ang_err"] += abs(wz - wz_act)
                acc["falls"] += int(terminated.sum().item())
                acc["slide"] += foot_slide()
                acc["pitch"] += abs(float(robot.data.projected_gravity_b[:, 0].item()))
                # cadence: rising edge of left-ankle contact
                net_f = contact.data.net_forces_w[:, ankle_ids, :]
                contact_now = bool((net_f[:, 0, :].norm(dim=-1) > 1.0).any().item())
                if contact_now and not prev_contact:
                    acc["contact_ons"] += 1
                    acc["speed_sum"] += abs(vx_act)
                prev_contact = contact_now
                # joint divergence check
                acc["max_joint_dev"] = max(
                    acc["max_joint_dev"],
                    float((robot.data.joint_pos - robot.data.default_joint_pos).abs().max().item()),
                )
                acc["max_joint_vel"] = max(acc["max_joint_vel"], float(robot.data.joint_vel.abs().max().item()))
                # obs is a TensorDict from the AMP wrapper; check each leaf tensor
                for value in obs.values():
                    if isinstance(value, torch.Tensor) and bool(torch.isnan(value).any().item()):
                        acc["nan_obs"] = True

        # summarize
        steps = acc["steps"]
        duration = steps * 0.02
        cadence = acc["contact_ons"] / duration  # steps per second
        stride = acc["speed_sum"] / acc["contact_ons"] if acc["contact_ons"] > 0 else float("nan")
        summary = {
            "duration_s": duration,
            "mean_lin_err": acc["lin_err"] / steps,
            "mean_ang_err": acc["ang_err"] / steps,
            "falls": acc["falls"],
            "mean_foot_slide": acc["slide"] / steps,
            "mean_pitch_proxy": acc["pitch"] / steps,
            "cadence_hz": cadence,
            "stride_m": stride,
            "max_joint_dev": acc["max_joint_dev"],
            "max_joint_vel": acc["max_joint_vel"],
            "nan_obs": acc["nan_obs"],
        }
        results[name] = summary
        print(
            f"  lin_err={summary['mean_lin_err']:.3f} ang_err={summary['mean_ang_err']:.3f} "
            f"falls={summary['falls']} slide={summary['mean_foot_slide']:.3f} "
            f"pitch={summary['mean_pitch_proxy']:.3f} cadence={cadence:.2f}Hz stride={stride:.2f}m "
            f"max_dev={summary['max_joint_dev']:.2f} max_vel={summary['max_joint_vel']:.2f} nan={summary['nan_obs']}"
        )

    # final report
    print("\n================ acceptance report ================")
    header = f"{'scenario':<18}{'lin_err':>9}{'ang_err':>9}{'falls':>7}{'slide':>8}{'pitch':>8}{'cad':>7}{'stride':>8}"
    print(header)
    for name, s in results.items():
        print(
            f"{name:<18}{s['mean_lin_err']:>9.3f}{s['mean_ang_err']:>9.3f}{s['falls']:>7d}"
            f"{s['mean_foot_slide']:>8.3f}{s['mean_pitch_proxy']:>8.3f}{s['cadence_hz']:>7.2f}{s['stride_m']:>8.2f}"
        )
    if "7_long_run" in results:
        print(
            "\nlong-run: "
            f"falls={results['7_long_run']['falls']} over {results['7_long_run']['duration_s']:.0f}s, "
            f"max_joint_dev={results['7_long_run']['max_joint_dev']:.2f}, "
            f"max_joint_vel={results['7_long_run']['max_joint_vel']:.2f}, "
            f"nan={results['7_long_run']['nan_obs']}"
        )

    report_path = os.path.join(log_dir, "acceptance_metrics.json")
    with open(report_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nmetrics saved to {report_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
