"""GMR -> HW8 AMP, with bounded-memory SMPL-X evaluation and CPU FK."""
from pathlib import Path
import os
import sys
import numpy as np

_adjacent = Path(__file__).resolve().parents[1]
GMR_ROOT = Path(os.environ.get('GMR_ROOT', str(_adjacent if (_adjacent / 'general_motion_retargeting').is_dir() else Path.home() / 'Desktop/github/GMR')))
sys.path.insert(0, str(GMR_ROOT))
KEY_LINKS = ('left_ankle_roll_link', 'right_ankle_roll_link',
             'left_wrist_yaw_link', 'right_wrist_yaw_link')
JOINT_NAMES = tuple(
    [f'{side}_{name}_joint' for side in ('left', 'right') for name in
     ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
    + [f'waist_{axis}_joint' for axis in ('yaw', 'roll', 'pitch')]
    + [f'{side}_{name}_joint' for side in ('left', 'right') for name in
       ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
        'wrist_roll', 'wrist_pitch', 'wrist_yaw')])


# Locomotion retargeting bounds, not manufacturer hardware ratings.
WRIST_LIMIT_DEG = {'roll': 45.0, 'pitch': 30.0, 'yaw': 20.0}


def add_wrist_constraints(retarget):
    """Prefer a neutral wrist and constrain it inside a locomotion range in IK."""
    import mink
    model = retarget.model
    bounds = mink.ConfigurationLimit(model)
    cost = np.zeros(model.nv)
    for side in ('left', 'right'):
        for axis, degrees in WRIST_LIMIT_DEG.items():
            joint = model.joint(f'{side}_wrist_{axis}_joint').id
            qadr = model.jnt_qposadr[joint]
            bounds.lower[qadr] = max(bounds.lower[qadr], -np.deg2rad(degrees))
            bounds.upper[qadr] = min(bounds.upper[qadr], np.deg2rad(degrees))
            cost[model.jnt_dofadr[joint]] = 5.0
        # Retain wrist position tracking; reduce pressure to copy hand orientation.
        for tasks in (retarget.human_body_to_task1, retarget.human_body_to_task2):
            tasks[f'{side}_wrist'].set_orientation_cost(0.5)
    neutral = mink.PostureTask(model, cost=cost)
    neutral.set_target(model.qpos0)
    retarget.tasks1.append(neutral)
    retarget.tasks2.append(neutral)
    retarget.ik_limits.append(bounds)
    return bounds


def validate(arrays):
    required = {'fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list'}
    if required - arrays.keys():
        raise ValueError(f'Missing fields: {required - arrays.keys()}')
    for name, a in arrays.items():
        if a.dtype.hasobject:
            raise ValueError(f'Object field: {name}')
        if a.dtype.kind in 'fci' and not np.isfinite(a).all():
            raise ValueError(f'Nonfinite field: {name}')
    if arrays['fps'].shape != () or float(arrays['fps']) <= 0:
        raise ValueError('fps must be a positive finite scalar')
    t = len(arrays['root_pos'])
    links = arrays['link_body_list']
    expected = {'root_pos': (t, 3), 'root_rot': (t, 4), 'dof_pos': (t, 29),
                'local_body_pos': (t, len(links), 3)}
    if t < 2:
        raise ValueError('At least two frames required')
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f'{name}: expected {shape}, got {arrays[name].shape}')
    if links.ndim != 1 or links.dtype.kind != 'U' or len(set(links)) != len(links):
        raise ValueError('link_body_list must contain unique Unicode names')
    if not set(KEY_LINKS).issubset(links):
        raise ValueError('Missing AMP key links')
    q = arrays['root_rot']
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5):
        raise ValueError('Nonunit quaternions')
    if np.any(np.sum(q[1:] * q[:-1], axis=1) < 0):
        raise ValueError('Quaternion sign discontinuity')
    if 'joint_names' in arrays and tuple(arrays['joint_names']) != JOINT_NAMES:
        raise ValueError('Incorrect joint order')


def human_frames(source, body_models, fps=30, batch_size=32):
    import torch
    import smplx
    from smplx.joint_names import JOINT_NAMES as HUMAN_NAMES
    from scipy.spatial.transform import Rotation, Slerp
    with np.load(source, allow_pickle=False) as archive:
        raw = {k: archive[k] for k in ('pose_body', 'root_orient', 'trans', 'betas', 'gender', 'mocap_frame_rate')}
    count = len(raw['trans'])
    source_fps = float(raw['mocap_frame_rate'])
    if count < 2 or not np.isfinite(source_fps) or source_fps <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid motion duration or frame rate')
    if raw['pose_body'].shape != (count, 63):
        raise ValueError('Expected AMASS SMPL-X pose_body (T,63)')
    duration = (count - 1) / source_fps
    # Uniform timestamps include both endpoints; preserve exact source duration.
    n = max(2, round(duration * fps) + 1)
    times = np.linspace(0, duration, n)
    old = np.arange(count) / source_fps
    pose = np.concatenate((raw['root_orient'], raw['pose_body']), axis=1).reshape(count, 22, 3)
    interpolated = np.stack([Slerp(old, Rotation.from_rotvec(pose[:, j]))(times).as_rotvec()
                             for j in range(22)], axis=1)
    trans = np.stack([np.interp(times, old, raw['trans'][:, j]) for j in range(3)], axis=1)
    gender = raw['gender'].item()
    if isinstance(gender, bytes):
        gender = gender.decode()
    betas = raw['betas'].reshape(-1)
    model = smplx.create(str(body_models), model_type='smplx', gender=str(gender),
                         use_pca=False, num_betas=len(betas), ext='pkl')
    frames = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            p = torch.tensor(interpolated[start:start+batch_size], dtype=torch.float32)
            b = len(p)
            out = model(betas=torch.tensor(betas, dtype=torch.float32)[None].expand(b, -1),
                        global_orient=p[:, 0], body_pose=p[:, 1:].reshape(b, 63),
                        transl=torch.tensor(trans[start:start+b], dtype=torch.float32),
                        left_hand_pose=torch.zeros(b, 45), right_hand_pose=torch.zeros(b, 45),
                        jaw_pose=torch.zeros(b, 3), leye_pose=torch.zeros(b, 3),
                        reye_pose=torch.zeros(b, 3), expression=torch.zeros(b, 10), return_full_pose=True)
            positions = out.joints.numpy()
            poses = out.full_pose.numpy().reshape(b, -1, 3)
            for i in range(b):
                rotations = []
                frame = {}
                for j, parent in enumerate(model.parents.tolist()):
                    r = Rotation.from_rotvec(poses[i, j])
                    if parent >= 0:
                        r = rotations[parent] * r
                    rotations.append(r)
                    frame[HUMAN_NAMES[j]] = (positions[i, j].copy(), r.as_quat(scalar_first=True))
                frames.append(frame)
    return frames, (n - 1) / duration, float(1.66 + .1 * betas[0])


def export_qpos(qpos, fps, model, path):
    import torch
    from general_motion_retargeting import KinematicsModel, ROBOT_XML_DICT
    if model.nq != 36 or tuple(model.joint(i).name for i in range(1, model.njnt)) != JOINT_NAMES:
        raise ValueError('Expected G1 29DoF in the HW8 joint order')
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    qpos[:, :2] -= qpos[0, :2]
    rot = qpos[:, [4, 5, 6, 3]]
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    for i in range(1, len(rot)):
        if rot[i] @ rot[i-1] < 0:
            rot[i] *= -1
    fk = KinematicsModel(str(ROBOT_XML_DICT['unitree_g1']), device='cpu')
    with torch.no_grad():
        identity = torch.zeros(len(qpos), 4); identity[:, 3] = 1
        local, _ = fk.forward_kinematics(torch.zeros(len(qpos), 3), identity,
                                        torch.tensor(qpos[:, 7:], dtype=torch.float32))
    arrays = dict(fps=np.asarray(fps, dtype=np.float64), root_pos=qpos[:, :3].astype('float32'),
                  root_rot=rot.astype('float32'), dof_pos=qpos[:, 7:].astype('float32'),
                  local_body_pos=local.numpy().astype('float32'),
                  link_body_list=np.asarray(fk.body_names, dtype=str),
                  joint_names=np.asarray(JOINT_NAMES, dtype=str))
    validate(arrays)
    path = Path(path)
    if path.suffix.lower() != '.npz':
        raise ValueError('Output must end in .npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)
    return arrays


def convert(source, output, body_models, fps=30):
    import torch
    from general_motion_retargeting import GeneralMotionRetargeting
    torch.set_num_threads(1)
    frames, aligned_fps, height = human_frames(source, body_models, fps)
    retarget = GeneralMotionRetargeting(src_human='smplx', tgt_robot='unitree_g1',
                                       actual_human_height=height, solver='daqp', verbose=False)
    import mink
    add_wrist_constraints(retarget)
    # Keep the hands clear of the torso while preserving the original GMR tasks.
    def collision_geoms(names):
        bodies = {retarget.model.body(name).id for name in names}
        return [g for g in range(retarget.model.ngeom)
                if retarget.model.geom_bodyid[g] in bodies
                and (retarget.model.geom_contype[g] or retarget.model.geom_conaffinity[g])]
    retarget.ik_limits.append(mink.CollisionAvoidanceLimit(
        retarget.model,
        [(collision_geoms(['torso_link']), collision_geoms(list(KEY_LINKS[2:])))],
        minimum_distance_from_collisions=.008, collision_detection_distance=.05))
    # Converge initial pose before recording frame zero, avoiding an initialization jump.
    for _ in range(15):
        retarget.retarget(frames[0])
    # Bound the total inter-frame change, across all of GMR's internal IK iterations.
    # Wrist limits suppress branch changes in the wrist orientation solution.
    frame_limit = mink.ConfigurationLimit(retarget.model)
    retarget.ik_limits.append(frame_limit)
    physical_lower = frame_limit.lower.copy()
    physical_upper = frame_limit.upper.copy()
    speed = np.asarray([6.0 if 'wrist' in name else 20.0 for name in JOINT_NAMES])
    qpos = []
    for frame in frames:
        previous = retarget.configuration.data.qpos.copy()
        frame_limit.lower[7:] = np.maximum(physical_lower[7:], previous[7:] - speed / aligned_fps)
        frame_limit.upper[7:] = np.minimum(physical_upper[7:], previous[7:] + speed / aligned_fps)
        qpos.append(retarget.retarget(frame))
    arrays = export_qpos(qpos, aligned_fps, retarget.model, output)
    print(f'Saved {output}: {len(frames)} frames, {aligned_fps:.6f} fps', flush=True)
    return arrays


def default_models():
    return GMR_ROOT / 'assets' / 'body_models'
