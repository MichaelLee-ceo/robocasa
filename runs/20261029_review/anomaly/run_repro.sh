#!/bin/bash
# Reproduce the drill-down rollouts without rendering (repro.py), at most 3 at a time.
# usage: ./run_repro.sh   (writes repro_out/*.txt)
cd "$(dirname "$0")"
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 MUJOCO_GL=egl
PY="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
run() { name=$1; shift; OBJS=$OBJS NSTEPS=${NSTEPS:-32} $PY repro.py "$@" > repro_out/$name.txt 2> repro_out/$name.err; echo "done $name $?"; }
OBJS=obj run KB_s17_t112_confirm run_confirm_KettleBoiling KettleBoiling_s17_single 112 0,1,2,3 0 v3 ramp=-0.8 &
OBJS=bowl run SIM_s1_t624_dense2 run_dense2_SteamInMicrowave SteamInMicrowave_s1_single 624 0,1,2,3 0 v4 ramp=-0.6 ramp=-0.4 &
OBJS=obj run KB_s19_t160_v2 run_v2/s0 KettleBoiling_s19_single 160 0,1,2,4 0 settled ramp=1 ramp=0.6 &
wait
OBJS=obj NSTEPS=16 run KB_s0_t0_v2 run_v2/s0 KettleBoiling_s0_single 0 0,1 0 settled ramp=0 &
OBJS=vegetable NSTEPS=16 run SIM_s12_t16_dense2 run_dense2_SteamInMicrowave SteamInMicrowave_s12_single 16 0,1 0 v4 v3 &
OBJS=meat0 run PIL_s20_t1920_cur run_PackIdenticalLunches PackIdenticalLunches_s20_single 1920 0,1,2,7 0 settled ramp=0.4 ramp=0.2 &
wait
OBJS=bowl NSTEPS=16 run SIM_s8_t800_fix3 run_SteamInMicrowave_fix3 SteamInMicrowave_s8_single 800 0,1 0 v3 ramp=-0.4 ramp=-0.6 &
wait
