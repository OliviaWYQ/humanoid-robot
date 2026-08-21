"""
HW2 Task 2: Humanoid Action Space — HoST Sim2Sim Student Starter

Complete the TODO blocks below to implement:
  1. Reading raw robot state from MuJoCo
  2. Building the single-step observation vector
  3. Maintaining the rolling observation history
  4. Converting policy output into current-pose incremental joint targets

See HW2_Task2_Action_Space.md for background and instructions.
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


def get_gravity_orientation(base_quat_wxyz):
    """Return gravity direction in the robot base frame (3,)."""
    qw = base_quat_wxyz[0]
    qx = base_quat_wxyz[1]
    qy = base_quat_wxyz[2]
    qz = base_quat_wxyz[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_joint_positions, current_joint_positions, kp, target_joint_velocities, current_joint_velocities, kd):
    """Position PD controller: returns joint torques."""
    return (target_joint_positions - current_joint_positions) * kp + (
        target_joint_velocities - current_joint_velocities
    ) * kd


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    package_root = os.path.dirname(os.path.abspath(config_path))
    package_root = os.path.dirname(package_root)  # configs/ -> package root

    def resolve_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(package_root, path)

    config["policy_path"] = resolve_path(config["policy_path"])
    config["xml_path"] = resolve_path(config["xml_path"])
    config["video_path"] = resolve_path(config.get("video_path", "outputs/simulation.mp4"))
    return config


def read_robot_state(mj_data, dof_pos_scale, dof_vel_scale, ang_vel_scale):
    """
    Read joint and base state from MuJoCo.

    Returns:
        current_joint_positions: (23,) unscaled joint angles in radians
        joint_positions_for_obs: (23,) scaled joint positions for the policy observation
        joint_velocities_for_obs: (23,) scaled joint velocities for the policy observation
        base_angular_velocity_obs: (3,) scaled base angular velocity in the base frame
        projected_gravity: (3,) gravity direction expressed in the base frame
    """
    # -------------------------------------------------------------------------
    # TODO 1: Read raw robot state from MuJoCo
    #
    # Hint:
    #   - MuJoCo stores the floating base in qpos[:7] and qvel[:6].
    #   - Joint positions start at qpos[7:], joint velocities at qvel[6:].
    #   - Base orientation quaternion is qpos[3:7] in [w, x, y, z] order.
    #   - Base angular velocity is qvel[3:6].
    #
    # Steps:
    #   1. Extract current_joint_positions from mj_data.qpos[7:].
    #   2. Extract raw joint velocities from mj_data.qvel[6:].
    #   3. Extract base_quat_wxyz from mj_data.qpos[3:7].
    #   4. Extract base_angular_velocity from mj_data.qvel[3:6].
    #   5. Apply observation scaling:
    #        joint_positions_for_obs = current_joint_positions * dof_pos_scale
    #        joint_velocities_for_obs = raw_joint_velocities * dof_vel_scale
    #        base_angular_velocity_obs = base_angular_velocity * ang_vel_scale
    #   6. Compute projected_gravity with get_gravity_orientation(base_quat_wxyz).
    #
    # Your code here:
    raise NotImplementedError("TODO 1: implement read_robot_state()")


def build_single_observation(
    base_angular_velocity_obs,
    projected_gravity,
    joint_positions_for_obs,
    joint_velocities_for_obs,
    previous_policy_action,
    action_scale,
):
    """
    Build one 76-dimensional observation vector.

    Layout (total = 76):
        [0:3]   base angular velocity          (3)
        [3:6]   projected gravity              (3)
        [6:29]  joint positions                (23)
        [29:52] joint velocities               (23)
        [52:75] previous policy action         (23)
        [75]    action scale scalar            (1)
    """
    # -------------------------------------------------------------------------
    # TODO 2: Build the single-step observation vector
    #
    # Verify: current_obs.shape == (76,)
    #
    # Your code here:
    raise NotImplementedError("TODO 2: implement build_single_observation()")


def update_observation_history(observation_history, current_obs, single_observation_dim, history_length):
    """
    Maintain a rolling buffer of the last `history_length` observations.

    Returns:
        observation_history: (single_observation_dim * history_length,) updated buffer
    """
    # -------------------------------------------------------------------------
    # TODO 3: Update the rolling observation history
    #
    # Hint:
    #   Drop the oldest single_observation_dim entries, then append current_obs:
    # Verify:
    #   observation_history.shape == (single_observation_dim * history_length,)
    #   For the default config: (76 * 6,) = (456,)
    #
    # Your code here:
    raise NotImplementedError("TODO 3: implement update_observation_history()")


def action_to_joint_targets(policy_action, current_joint_positions, action_scale):
    """
    Convert policy output to joint position targets using the HoST incremental action space.

    HoST formulation:
        target_joint_positions = current_joint_positions + action_scale * policy_action

    Compare with the standard residual formulation:
        target_joint_positions = default_joint_positions + action_scale * policy_action
    """
    # -------------------------------------------------------------------------
    # TODO 4: Convert policy output into current-pose incremental joint targets
    #
    # Hint:
    #
    # Return:
    #   target_joint_positions with shape (23,)
    #
    # Your code here:
    raise NotImplementedError("TODO 4: implement action_to_joint_targets()")


def main():
    package_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(package_root, "configs", "g1.yaml")
    config = load_config(config_path)

    print("Config:", config_path)
    print("Policy:", config["policy_path"])
    print("Robot model:", config["xml_path"])

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    joint_kp = np.array(config["kps"], dtype=np.float32)
    joint_kd = np.array(config["kds"], dtype=np.float32)
    default_joint_positions = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]

    num_actions = config["num_actions"]
    single_observation_dim = config["single_observation_dim"]
    history_length = config["history_length"]

    torque_limits = np.array(config["torque_limits"], dtype=np.float32)

    save_video = config.get("save_video", False)
    video_path = config["video_path"]
    video_fps = config.get("video_fps", 50)
    video_width = config.get("video_width", 1280)
    video_height = config.get("video_height", 720)

    # -------------------------------------------------------------------------
    # Control state
    previous_policy_action = np.zeros(num_actions, dtype=np.float32)
    target_joint_positions = default_joint_positions.copy()
    observation_history = np.zeros(
        single_observation_dim * history_length, dtype=np.float32
    )

    # Load MuJoCo model and policy
    mj_model = mujoco.MjModel.from_xml_path(config["xml_path"])
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = simulation_dt
    policy = torch.jit.load(config["policy_path"])

    # Optional video recording
    video_frame_stride = max(1, round((1.0 / video_fps) / simulation_dt))
    renderer = None
    camera = None
    video_writer = None
    video_frame_count = 0
    if save_video:
        import imageio

        mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, video_width)
        mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, video_height)
        renderer = mujoco.Renderer(mj_model, height=video_height, width=video_width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(mj_model, camera)
        camera.distance = 2.5
        camera.azimuth = 135
        camera.elevation = -20
        camera.lookat[:] = [0.0, 0.0, 0.8]
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=video_fps, macro_block_size=1)

    sim_step_counter = 0

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < simulation_duration:
                step_start = time.time()

                # Low-level PD tracking (runs every physics step)
                joint_torques = pd_control(
                    target_joint_positions,
                    mj_data.qpos[7:],
                    joint_kp,
                    np.zeros_like(joint_kd),
                    mj_data.qvel[6:],
                    joint_kd,
                )
                joint_torques = np.clip(joint_torques, -torque_limits, torque_limits)
                mj_data.ctrl[:] = joint_torques
                mujoco.mj_step(mj_model, mj_data)

                sim_step_counter += 1
                if sim_step_counter % control_decimation == 0:
                    # ---------------------------------------------------------
                    # Policy control loop (50 Hz)
                    # ---------------------------------------------------------
                    (
                        current_joint_positions,
                        joint_positions_for_obs,
                        joint_velocities_for_obs,
                        base_angular_velocity_obs,
                        projected_gravity,
                    ) = read_robot_state(
                        mj_data, dof_pos_scale, dof_vel_scale, ang_vel_scale
                    )

                    current_obs = build_single_observation(
                        base_angular_velocity_obs,
                        projected_gravity,
                        joint_positions_for_obs,
                        joint_velocities_for_obs,
                        previous_policy_action,
                        action_scale,
                    )

                    observation_history = update_observation_history(
                        observation_history,
                        current_obs,
                        single_observation_dim,
                        history_length,
                    )

                    obs_tensor = torch.from_numpy(observation_history).unsqueeze(0).float()
                    policy_action = policy(obs_tensor).detach().numpy().squeeze()
                    previous_policy_action = policy_action.copy()

                    target_joint_positions = action_to_joint_targets(
                        policy_action,
                        current_joint_positions,
                        action_scale,
                    )

                if save_video and sim_step_counter % video_frame_stride == 0:
                    renderer.update_scene(mj_data, camera=camera)
                    video_writer.append_data(renderer.render())
                    video_frame_count += 1

                viewer.sync()

                time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        if renderer is not None:
            renderer.close()
        if video_writer is not None:
            video_writer.close()
            if video_frame_count > 0:
                print(f"Saved video ({video_frame_count} frames) to: {video_path}")


if __name__ == "__main__":
    main()
