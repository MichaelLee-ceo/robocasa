#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=disable
cd /tmp && nice -n 19 ionice -c3 python -I /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/faucet_spout.py /home/michaellee/mclee/robocasa/runs/_handoff/verify_asus/faucet_spout.json  "PreSoakPan:/home/michaellee/mclee/robocasa/runs/20261014_asus_seen5:4,8,10,11,16,17,22,23,24,27,28,0,1,2,3,5"  "WashLettuce:/home/michaellee/mclee/robocasa/runs/20261014_asus_seen11:4,8,11,15,20,10,22,1,7,9,12,13"  "RinseSinkBasin:/home/michaellee/mclee/robocasa/runs/20261014_asus_seen11:3,4,10,15,17,22,26,8,0,1,2,6,9" 2>&1 | grep -vE 'WARNING|INFO'
echo FS_DONE
