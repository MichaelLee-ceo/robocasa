#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 NO_ALBUMENTATIONS_UPDATE=1
cd /tmp && nice -n 10 python -I /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/raw_coffee_kettle.py "$@" 2>&1 | grep -vE 'WARNING|INFO|Creating'
echo RAW_DONE
