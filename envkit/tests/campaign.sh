#!/bin/bash
# Freeze every scene (16 Composite-Seen tasks x seeds 0-29) and check it, N jobs at a time on one GPU.
# usage: setsid nohup ./campaign.sh <gpu> <jobs> > results/campaign.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
GPU=${1:-1}; N=${2:-16}
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
mkdir -p results
TASKS="KettleBoiling PreSoakPan StackBowlsCabinet SteamInMicrowave PrepareCoffee ScrubCuttingBoard WashLettuce RinseSinkBasin LoadDishwasher StoreLeftoversInBowl SetUpCuttingStation StirVegetables SearingMeat PackIdenticalLunches DeliverStraw GetToastedBread"
for t in $TASKS; do for s in $(seq 0 29); do echo "$t $s"; done; done | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$N" -L 1 bash -c "$PY run_tests.py freeze \$0 \$1 2>/dev/null | grep '^{\"cmd\"' >> results/freeze.jsonl; $PY run_tests.py check \$0 \$1 2>/dev/null | grep '^{\"cmd\"' >> results/check.jsonl"
echo "done $(date)"
