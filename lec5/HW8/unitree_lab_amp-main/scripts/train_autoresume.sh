#!/usr/bin/env bash
# AMP training with liveness watchdog and crash auto-resume.
#
# NOTE (this machine, measured 2026-10-23): 2-GPU distributed mode deadlocks
# within ~1-2 minutes (NCCL collective stall, GPUs spin at 100%) regardless of
# envs per rank (1024 and 512 both tested; CUDA_LAUNCH_BLOCKING=1 does not
# help), so single-GPU is the default. NUM_GPUS can be raised to 2 on machines
# without this issue.
#
# Behavior: runs training, kills stalls after 300 s without a new iteration
# (faster than the 600 s NCCL watchdog), and on failure relaunches from the
# latest checkpoint. Up to MAX_ATTEMPTS attempts.
#
# TOTAL_ITERATIONS is the FINAL target iteration. This runner treats
# --max_iterations as "iterations to run from the current one" (learn() uses
# total = start + num_learning_iterations), so on resume we pass the REMAINING
# iterations (TOTAL_ITERATIONS - checkpoint_iter), not the full target.
# NOTE: no `set -u` here - the conda activation chain (Isaac Sim's
# setup_conda_env.sh) references unset vars like $ZSH_VERSION.
:

REPO=/home/star/Desktop/humanoid-robot/lec5/HW8/unitree_lab_amp-main
LOG=/tmp/train_auto.log
LOG_ROOT="$REPO/logs/rsl_rl_amp/unitree_g1_29dof_amp_walk_to_run"
MAX_ATTEMPTS=12
NUM_GPUS=1
NUM_ENVS=1024
TOTAL_ITERATIONS=50000
MAX_ITERATIONS="$TOTAL_ITERATIONS"
RESUME_ARGS=""

cd "$REPO" || exit 1
source /home/star/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Pick the newest checkpoint file across ALL run dirs (a newer run dir may
# hold only a stale model_0.pt) and compute the remaining iterations to reach
# TOTAL_ITERATIONS. Checkpoint names are model_<iter>.pt by construction.
compute_resume() {
  CKPT=$(ls -t "$LOG_ROOT"/*/model_*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    RUN=$(basename "$(dirname "$CKPT")")
    CKPT_ITER=$(basename "$CKPT" .pt); CKPT_ITER=${CKPT_ITER#model_}
    MAX_ITERATIONS=$((TOTAL_ITERATIONS - CKPT_ITER))
    [ "$MAX_ITERATIONS" -lt 0 ] && MAX_ITERATIONS=0
    RESUME_ARGS="--resume --load_run $RUN --checkpoint $(basename "$CKPT")"
    echo "found checkpoint $CKPT (iter $CKPT_ITER); $MAX_ITERATIONS iterations remaining to $TOTAL_ITERATIONS" | tee -a "$LOG"
  else
    MAX_ITERATIONS="$TOTAL_ITERATIONS"
    RESUME_ARGS=""
    echo "no checkpoint yet, fresh start for $MAX_ITERATIONS iterations" | tee -a "$LOG"
  fi
}

compute_resume
if [ "$MAX_ITERATIONS" -le 0 ]; then
  echo "already at/over target $TOTAL_ITERATIONS - nothing to do" | tee -a "$LOG"
  exit 0
fi

ATTEMPT=0
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  : > "$LOG"
  echo "=== attempt $ATTEMPT (gpus=$NUM_GPUS, envs=$NUM_ENVS, max_iterations=$MAX_ITERATIONS, resume='$RESUME_ARGS') ===" | tee -a "$LOG"

  if [ "$NUM_GPUS" -ge 2 ]; then
    python -m torch.distributed.run --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      scripts/rsl_rl/train.py --headless --distributed \
      --task Unitree-G1-29dof-AMP-WalkToRun --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
      $RESUME_ARGS >> "$LOG" 2>&1 &
  else
    python scripts/rsl_rl/train.py --headless \
      --task Unitree-G1-29dof-AMP-WalkToRun --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERATIONS" \
      $RESUME_ARGS >> "$LOG" 2>&1 &
  fi
  TRAIN_PID=$!

  # Liveness watchdog: kill if no new iteration line for 300 s.
  PREV=0
  STALL=0
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 30
    CUR=$(grep -c "Iteration time" "$LOG" 2>/dev/null || true)
    if [ "$CUR" -le "$PREV" ]; then
      STALL=$((STALL + 1))
      if [ "$STALL" -ge 10 ]; then
        echo "[watchdog] $((STALL * 30)) s without progress - killing run" | tee -a "$LOG"
        pkill -f "torch.distributed.run" 2>/dev/null
        pkill -f "scripts/rsl_rl/train.py" 2>/dev/null
        break
      fi
    else
      STALL=0
    fi
    PREV=$CUR
  done

  wait "$TRAIN_PID"
  RC=$?
  if [ "$RC" -eq 0 ]; then
    echo "training finished successfully (target $TOTAL_ITERATIONS)" | tee -a "$LOG"
    exit 0
  fi

  # Wait for any stragglers to fully exit before relaunching (kit teardown).
  for _ in $(seq 1 10); do
    pgrep -f "scripts/rsl_rl/train.py" >/dev/null || break
    sleep 3
  done
  pkill -9 -f "scripts/rsl_rl/train.py" 2>/dev/null
  pkill -9 -f "torch.distributed.run" 2>/dev/null
  sleep 5

  compute_resume
  if [ "$MAX_ITERATIONS" -le 0 ]; then
    echo "reached target $TOTAL_ITERATIONS after attempt $ATTEMPT failed - done" | tee -a "$LOG"
    exit 0
  fi
  echo "attempt $ATTEMPT failed (rc=$RC); will resume with $MAX_ITERATIONS remaining" | tee -a "$LOG"
done

echo "max attempts ($MAX_ATTEMPTS) reached - giving up" | tee -a "$LOG"
exit 1
