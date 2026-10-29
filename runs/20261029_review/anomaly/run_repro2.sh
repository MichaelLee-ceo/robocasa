#!/bin/bash
# Second drill-down batch: candidates at grasp onsets (class C) with the settled ramp vs the ramp estimated from the
# main run's finger gap (r = 1 - gap / 0.04, rounded to the ramp's 0.2 steps). usage: ./run_repro2.sh
cd "$(dirname "$0")"
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 MUJOCO_GL=egl
PY="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
run() { name=$1; shift; OBJS=$OBJS NSTEPS=${NSTEPS:-32} $PY repro.py "$@" > repro_out/$name.txt 2> repro_out/$name.err; echo "done $name $?"; }
OBJS=obj NSTEPS=24 run KB_s17_t112_confirm_all run_confirm_KettleBoiling KettleBoiling_s17_single 112 0,1,2,3,4,5,6,7 0 v3 ramp=-0.8 ramp=-0.6 &
OBJS=obj NSTEPS=24 run KB_s19_t96_dense run_dense_KettleBoiling KettleBoiling_s19_single 96 0,1,2,3,4,5,6,7 0 basefix ramp=-0.4 &
OBJS=meat1 NSTEPS=24 run PIL_s11_t512_dense run_dense_PackIdenticalLunches PackIdenticalLunches_s11_single 512 0,1,2,3,4,5,6,7 0 v3 ramp=-0.8 &
wait
OBJS=meat0 NSTEPS=24 run PIL_s21_t96_dense run_dense_PackIdenticalLunches PackIdenticalLunches_s21_single 96 0,1,2,3,4,5,6,7 0 v3 ramp=-0.8 &
wait
