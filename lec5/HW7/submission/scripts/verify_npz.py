"""Independent MuJoCo FK and real HW8 loader checks; writes quality diagnostics."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from gmr_npz import validate, JOINT_NAMES, KEY_LINKS, WRIST_LIMIT_DEG


def wrist_quality(dof):
    result = {}
    for side in ('left', 'right'):
        for axis, limit in WRIST_LIMIT_DEG.items():
            name = f'{side}_wrist_{axis}_joint'
            values = np.rad2deg(dof[:, JOINT_NAMES.index(name)])
            assert np.max(np.abs(values)) <= limit + 1e-3, f'{name}: wrist range exceeded'
            near = np.abs(values) >= limit - 1.0
            padded = np.r_[False, near, False].astype(int)
            starts = np.where(np.diff(padded) == 1)[0]
            ends = np.where(np.diff(padded) == -1)[0]
            run = max(ends - starts, default=0)
            assert np.mean(near) < .25, f'{name}: sustained wrist limit saturation'
            result[name] = dict(min_deg=float(values.min()), max_deg=float(values.max()),
                                near_limit_fraction=float(np.mean(near)), longest_near_limit_frames=int(run))
    return result


def check(path):
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from general_motion_retargeting import ROBOT_XML_DICT
    with np.load(path, allow_pickle=False) as z:
        a = dict(z)
    validate(a)
    wrists = wrist_quality(a['dof_pos'])
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT['unitree_g1']))
    data = mj.MjData(model)
    assert tuple(model.joint(i).name for i in range(1, model.njnt)) == JOINT_NAMES
    ids = [model.body(str(n)).id for n in a['link_body_list']]
    feet = [model.body(n).id for n in KEY_LINKS[:2]]
    foot_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in feet
                  and model.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE]
    errors, heights, foot_pos, collision_depth = [], [], [], []
    for pos, rot, dof, local in zip(a['root_pos'], a['root_rot'], a['dof_pos'], a['local_body_pos']):
        data.qpos[:] = np.r_[pos, rot[[3, 0, 1, 2]], dof]
        mj.mj_forward(model, data)
        expected = (data.xpos[ids] - pos) @ Rotation.from_quat(rot).as_matrix()
        errors.append(float(np.max(np.abs(expected - local))))
        heights.append([min(data.geom_xpos[g, 2] - model.geom_size[g, 0]
                            for g in foot_geoms if model.geom_bodyid[g] == f) for f in feet])
        foot_pos.append(data.xpos[feet].copy())
        collision_depth.append(max([max(0., -c.dist) for c in data.contact
                                    if model.geom_bodyid[c.geom1] != 0 and model.geom_bodyid[c.geom2] != 0] or [0.]))
    assert max(errors) < 1e-5, f'FK mismatch {max(errors)}'
    lower, upper = model.jnt_range[1:].T
    violation = np.maximum(lower - a['dof_pos'], a['dof_pos'] - upper).clip(min=0)
    assert violation.max() < 1e-4, f'Joint limit violation {violation.max()}'
    velocity = np.abs(np.diff(a['dof_pos'], axis=0)) * float(a['fps'])
    wrist_indices = [i for i, n in enumerate(JOINT_NAMES) if 'wrist' in n]
    wrist_speed = float(velocity[:, wrist_indices].max())
    assert wrist_speed <= 6.0001, f'Wrist velocity discontinuity: {wrist_speed}'
    assert float(velocity.max()) <= 20.0001, 'Frame velocity bound violated'
    heights = np.asarray(heights)
    speed = np.linalg.norm(np.diff(foot_pos, axis=0)[:, :, :2], axis=-1) * float(a['fps'])
    contact = (heights[:-1] < .04) & (heights[1:] < .04)
    slip = speed[contact]
    return dict(file=path.name, frames=len(a['root_pos']), fps=float(a['fps']),
                duration_seconds=(len(a['root_pos'])-1)/float(a['fps']),
                local_body_shape=list(a['local_body_pos'].shape), fk_max_error_m=max(errors),
                joint_limit_violation_rad=float(violation.max()),
                max_joint_speed_rad_s=float(velocity.max()), max_wrist_speed_rad_s=wrist_speed,
                min_sole_height_m=float(heights.min()),
                max_lower_sole_height_m=float(heights.min(axis=1).max()),
                near_ground_foot_speed_median_m_s=float(np.median(slip)) if len(slip) else None,
                near_ground_foot_speed_p95_m_s=float(np.percentile(slip,95)) if len(slip) else None,
                self_collision_max_depth_m=max(collision_depth),
                self_collision_frame_fraction=float(np.mean(np.asarray(collision_depth) > .005)),
                wrist_quality=wrists,
                heading_change_deg=float(np.rad2deg(np.diff(np.unwrap(Rotation.from_quat(a['root_rot']).as_euler('xyz')[:, 2]))).sum()),
                format_fk_joint_limits_passed=True)


def amp_check(folder, loader):
    import torch
    spec = importlib.util.spec_from_file_location('hw8_motion_dataset', loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.MotionDatasetCfg(motion_dir=str(folder), joint_names=JOINT_NAMES,
                                  source_joint_names=JOINT_NAMES, key_link_names=KEY_LINKS,
                                  history_steps=3, step_dt=.02,
                                  clip_weights={p.stem: 1. for p in folder.glob('*.npz')})
    dataset = module.MotionDataset(cfg, device='cpu')
    torch.manual_seed(7)
    batch = dataset.sample(128)
    assert torch.isfinite(batch).all()
    assert len(dataset.clips) == len(list(folder.glob('*.npz')))
    return dict(clips=[c.name for c in dataset.clips], sample_shape=list(batch.shape),
                frame_dim=dataset.frame_dim, finite=True, loader=str(loader))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    p.add_argument('--amp_loader', type=Path)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    paths = sorted(a.folder.glob('*.npz'))
    if not paths:
        p.error('No NPZ files')
    result = {'motions': [check(f) for f in paths]}
    if a.amp_loader:
        result['amp'] = amp_check(a.folder, a.amp_loader)
    a.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
