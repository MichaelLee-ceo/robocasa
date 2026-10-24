#!/bin/bash
# Remaining scene generation (detached): one process at a time, same as before.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 NO_ALBUMENTATIONS_UPDATE=1
cd /tmp
for x in "PrepareCoffee scenes_seen5_v2_s30" "StackBowlsCabinet scenes_seen5_v2_s30" "SteamInMicrowave scenes_seen5_v2_s30" \
         "GetToastedBread scenes_seen11_v2" "DeliverStraw scenes_seen11_v2" "SearingMeat scenes_seen11_v2" "PackIdenticalLunches scenes_seen11_v2" \
         "StirVegetables scenes_seen11_v2" "SetUpCuttingStation scenes_seen11_v2" "StoreLeftoversInBowl scenes_seen11_v2" "LoadDishwasher scenes_seen11_v2" "ScrubCuttingBoard scenes_seen11_v2"; do
  set -- $x
  nice -n 10 python -I $ROBOCASA_ROOT/runs/scripts/make_scenes.py --tasks $1 --seeds 30 --out $ROBOCASA_ROOT/runs/$2 > $ROBOCASA_ROOT/runs/_handoff/verify_asus/scenegen/$1.log 2>&1
  echo "$(date +%T) $1 exit=$?"
done
