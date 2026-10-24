#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=disable
cd /tmp && nice -n 19 ionice -c3 python -I /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/searing_state.py /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/searing_state.json 2>&1 | grep -vE 'WARNING|INFO'
echo SS_DONE
