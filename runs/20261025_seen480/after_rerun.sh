#!/bin/bash
# Waits for the rerun to finish (run.out "done:"), then runs the stage analysis on all episodes with 32 workers.
cd /home/michaellee/mclee/robocasa/runs/20261025_seen480
until grep -q "done:" run.out; do sleep 30; done
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export PIP_CONFIG_FILE=/dev/null MUJOCO_GL=egl OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
echo "$(date +%T) rerun done, starting analysis" >> after_rerun.log
python -I analyze_stages.py run analysis 32 >> analysis_all.log 2>&1
echo "$(date +%T) analysis finished: $(grep -c '"tag"' analysis/stages.jsonl) lines, $(grep -c '"error"' analysis/stages.jsonl) errors" >> after_rerun.log
