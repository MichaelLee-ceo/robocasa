#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
cd /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/seen11
nice -n 19 ionice -c3 python -I diag.py "$@"
