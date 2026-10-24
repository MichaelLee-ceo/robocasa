#!/bin/bash
# usage: raw_generic_run.sh <script.py> <out.json>   (CPU only, no renderer)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=disable
cd /tmp && nice -n 19 ionice -c3 python -I "$1" "${@:2}" 2>&1 | grep -vE 'WARNING|INFO'
echo RAW_DONE
