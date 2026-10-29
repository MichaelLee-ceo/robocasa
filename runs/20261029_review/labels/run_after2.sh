#!/bin/bash
# Second queue (one process slot): waits for run_after.sh, then more pattern replays and the render check.
cd "$(dirname "$0")"
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 MUJOCO_GL=egl
PY="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
while pgrep -f "/bin/bash ./run_after.sh" > /dev/null; do sleep 20; done
echo "== compare_main PIL s5 $(date)"
$PY compare_main.py PackIdenticalLunches PackIdenticalLunches_s5_single 128,320,352,368,384,512,608 2>&1 | grep -v "WARNING\|INFO\|mimicgen\|Creating"
echo "== render_check $(date)"; $PY render_check.py 2>&1 | grep -v "WARNING\|INFO"
echo "== done $(date)"
