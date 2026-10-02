import argparse
from gmr_npz import convert, default_models


def main():
    p = argparse.ArgumentParser(description='SMPL-X to complete G1 AMP NPZ')
    p.add_argument('--smplx_file', required=True)
    p.add_argument('--save_path', required=True)
    p.add_argument('--robot', choices=['unitree_g1'], default='unitree_g1')
    p.add_argument('--body_models', default=str(default_models()))
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--rate_limit', action='store_true')
    a = p.parse_args()
    convert(a.smplx_file, a.save_path, a.body_models, a.fps)
    if not a.headless:
        from vis_robot_motion_npz import play
        play(a.save_path, rate_limit=a.rate_limit)

if __name__ == '__main__':
    main()
