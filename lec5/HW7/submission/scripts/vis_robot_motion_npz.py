"""Playback via GMR's MuJoCo viewer, or render an MP4 without a window."""
import argparse
from pathlib import Path
import numpy as np
from gmr_npz import validate


def play(path, rate_limit=True, video=None, headless=False, loop=False):
    import mujoco as mj
    from general_motion_retargeting import RobotMotionViewer, ROBOT_XML_DICT
    with np.load(path, allow_pickle=False) as z:
        arrays = dict(z)
    validate(arrays)
    fps = float(arrays['fps'])
    if not headless:
        viewer = RobotMotionViewer('unitree_g1', motion_fps=fps, record_video=bool(video), video_path=video)
        try:
            while True:
                for pos, rot, dof in zip(arrays['root_pos'], arrays['root_rot'], arrays['dof_pos']):
                    if not viewer.viewer.is_running():
                        return
                    viewer.step(pos, rot[[3, 0, 1, 2]], dof, rate_limit=rate_limit)
                if not loop:
                    break
        finally:
            viewer.close()
        return
    if not video:
        raise ValueError('--headless requires --video')
    import imageio.v2 as imageio
    # Add a visible ground plane without changing the retargeting model.
    spec = mj.MjSpec.from_file(str(ROBOT_XML_DICT['unitree_g1']))
    if not any(g.type == mj.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_PLANE, size=[200, 200, .1], rgba=[.75,.78,.8,1])
    spec.worldbody.add_light(pos=[0,-3,5], dir=[0,0,-1])
    model = spec.compile()
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    data = mj.MjData(model)
    cam = mj.MjvCamera()
    cam.distance = 3.2; cam.elevation = -15; cam.azimuth = 135
    Path(video).parent.mkdir(parents=True, exist_ok=True)
    with mj.Renderer(model, height=720, width=960) as renderer, imageio.get_writer(video, fps=fps) as writer:
        for pos, rot, dof in zip(arrays['root_pos'], arrays['root_rot'], arrays['dof_pos']):
            data.qpos[:] = np.r_[pos, rot[[3,0,1,2]], dof]
            mj.mj_forward(model, data)
            cam.lookat[:] = pos + np.array([0, 0, .1])
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    print(f'Video saved: {video}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--robot_motion_path', required=True)
    p.add_argument('--robot', choices=['unitree_g1'], default='unitree_g1')
    p.add_argument('--video')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--loop', action='store_true')
    a = p.parse_args()
    play(a.robot_motion_path, video=a.video, headless=a.headless, loop=a.loop)
