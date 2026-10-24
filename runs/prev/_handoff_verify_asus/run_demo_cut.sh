#!/bin/bash
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
V=/home/michaellee/mclee/robocasa/runs/_handoff/verify_asus
bash $V/raw_generic_run.sh $V/demo_cut_start.py $V/demo_cut_start.json /home/michaellee/mclee/robocasa/datasets/v1.0/target/composite/SetUpCuttingStation/20250817/lerobot
