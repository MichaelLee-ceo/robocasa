#!/bin/bash
# Sequential follow-up checks in one process slot (the label check uses two). Waits for pil_assign.py to finish.
cd "$(dirname "$0")"
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 MUJOCO_GL=egl
PY="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
while pgrep -f "python -I pil_assign.py" > /dev/null; do sleep 20; done
echo "== prev_ok_mismatch $(date)"; $PY prev_ok_mismatch.py 2>&1 | grep -v "WARNING\|INFO\|mimicgen\|Creating"
echo "== grip_against $(date)"; $PY grip_against.py 2>&1 | grep -v "WARNING\|INFO\|mimicgen\|Creating"
echo "== resim exactness $(date)"
$PY resim.py run_SteamInMicrowave_fix3 SteamInMicrowave_s14_single 1280 1 0 2>&1 | grep "^{"
$PY resim.py run_dense2_SteamInMicrowave SteamInMicrowave_s9_single 816 0 0 2>&1 | grep "^{"
$PY resim.py run_dense_KettleBoiling KettleBoiling_s26_single 112 0 0 2>&1 | grep "^{"
echo "== compare_main $(date)"
$PY compare_main.py PackIdenticalLunches PackIdenticalLunches_s16_single 320,640,960 2>&1 | grep -v "WARNING\|INFO\|mimicgen\|Creating"
$PY compare_main.py KettleBoiling KettleBoiling_s17_single 80,96,112,128 2>&1 | grep -v "WARNING\|INFO\|mimicgen\|Creating"
$PY compare_main.py SteamInMicrowave SteamInMicrowave_s8_single 928,960,992,1056,1072,1088 2>&1 | grep -v "WARNING\|INFO\|mimicgen\|Creating"
$PY compare_main.py PackIdenticalLunches PackIdenticalLunches_s18_single 1312,1344,1376 2>&1 | grep -v "WARNING\|INFO\|mimicgen\|Creating"
echo "== frame_scan $(date)"; $PY frame_scan.py 60 2>&1 | grep -v "WARNING\|INFO"
echo "== done $(date)"
