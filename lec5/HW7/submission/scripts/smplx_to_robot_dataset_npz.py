import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from gmr_npz import convert, default_models, validate
import numpy as np


def job(spec):
    source, target, models, fps, override = spec
    try:
        if target.exists() and not override:
            with np.load(target, allow_pickle=False) as z:
                validate(dict(z))
            return dict(source=str(source), output=str(target), status='validated_existing')
        convert(source, target, models, fps)
        return dict(source=str(source), output=str(target), status='generated')
    except Exception as e:
        return dict(source=str(source), output=str(target), status='failed', error=str(e))


def main():
    p = argparse.ArgumentParser(description='Batch SMPL-X to complete G1 AMP NPZ')
    p.add_argument('--src_folder', type=Path, required=True)
    p.add_argument('--tgt_folder', type=Path, required=True)
    p.add_argument('--robot', choices=['unitree_g1'], default='unitree_g1')
    p.add_argument('--body_models', default=str(default_models()))
    p.add_argument('--num_cpus', type=int, default=1)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--override', action='store_true')
    p.add_argument('--pattern', default='*_stageii.npz')
    a = p.parse_args()
    if a.num_cpus < 1 or a.fps <= 0:
        p.error('num_cpus and fps must be positive')
    sources = sorted(a.src_folder.rglob(a.pattern))
    if not sources:
        p.error('No matching source files')
    jobs = [(s, a.tgt_folder / s.relative_to(a.src_folder), a.body_models, a.fps, a.override) for s in sources]
    if a.num_cpus == 1:
        results = list(map(job, jobs))
    else:
        with concurrent.futures.ProcessPoolExecutor(a.num_cpus, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(job, jobs))
    a.tgt_folder.mkdir(parents=True, exist_ok=True)
    (a.tgt_folder / 'manifest.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
