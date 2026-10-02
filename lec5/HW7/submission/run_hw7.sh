#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../.venv/bin/python"
GMR_ROOT="${GMR_ROOT:-/Users/mac/Desktop/github/GMR}"
export GMR_ROOT
"$PY" "$HERE/scripts/smplx_to_robot_dataset_npz.py" \
  --src_folder "$HERE/input" --tgt_folder "$HERE/output" \
  --body_models "$GMR_ROOT/assets/body_models" --num_cpus 1 "$@"
"$PY" -m unittest discover -s "$HERE/tests" -v
"$PY" "$HERE/scripts/verify_npz.py" "$HERE/output" \
  --amp_loader "$HERE/../../HW8/unitree_lab_amp-main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/amp/motion_dataset.py" \
  --report "$HERE/output/validation.json"
for motion in "$HERE"/output/*.npz; do
  "$PY" "$HERE/scripts/vis_robot_motion_npz.py" \
    --robot_motion_path "$motion" --headless --video "${motion%.npz}.mp4"
done
