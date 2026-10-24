#!/bin/bash
# Scene generation, remaining tasks in the run order set on 2026-10-14 15:37 (one process at a time).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 NO_ALBUMENTATIONS_UPDATE=1 PIP_CONFIG_FILE=/dev/null
cd /tmp
for T in SearingMeat DeliverStraw PackIdenticalLunches StirVegetables SetUpCuttingStation StoreLeftoversInBowl LoadDishwasher ScrubCuttingBoard; do
  [ -f /home/michaellee/mclee/robocasa_rc/STOP ] && { echo "$(date +%T) STOP"; exit 0; }
  [ "$(ls $ROBOCASA_ROOT/runs/scenes_seen11_v2 | grep -c "^${T}_s")" -eq 30 ] && continue
  nice -n 10 python -I $ROBOCASA_ROOT/runs/scripts/make_scenes.py --tasks $T --seeds 30 --out $ROBOCASA_ROOT/runs/scenes_seen11_v2 > $ROBOCASA_ROOT/runs/_handoff/verify_asus/scenegen/$T.log 2>&1
  echo "$(date +%T) $T exit=$?"
done
