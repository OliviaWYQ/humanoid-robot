#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../.venv/bin/python"
MOTION="${1:-$HERE/output/B3_-_walk1_stageii.npz}"
if [[ "$(uname -s)" == Darwin ]]; then
  PYLIB="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
  DYLD_FALLBACK_LIBRARY_PATH="$PYLIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}" \
    "$PY" "$HERE/../.venv/bin/mjpython" "$HERE/scripts/vis_robot_motion_npz.py" \
    --robot_motion_path "$MOTION" --loop
else
  "$PY" "$HERE/scripts/vis_robot_motion_npz.py" --robot_motion_path "$MOTION" --loop
fi
