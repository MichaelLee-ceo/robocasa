#!/bin/bash
# GR00T policy tests (run_tests.py policy) for the 16 tasks x seeds SEED0..SEED1, N workers spread over the server ports.
# usage: setsid nohup ./policy_tests.sh <gpu> <jobs> <seed0> <seed1> "<ports>" > results/policy_tests.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
GPU=${1:-1}; N=${2:-8}; S0=${3:-0}; S1=${4:-2}; PORTS=(${5:-5581 5582})
PY="${GROOT_PY:-$(cd ../../.. && pwd)/envs/groot/bin/python} -I"
mkdir -p results/policy
TASKS="KettleBoiling PreSoakPan StackBowlsCabinet SteamInMicrowave PrepareCoffee ScrubCuttingBoard WashLettuce RinseSinkBasin LoadDishwasher StoreLeftoversInBowl SetUpCuttingStation StirVegetables SearingMeat PackIdenticalLunches DeliverStraw GetToastedBread"
i=0
for s in $(seq $S0 $S1); do for t in $TASKS; do echo "$t $s ${PORTS[$((i % ${#PORTS[@]}))]}"; i=$((i + 1)); done; done | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$N" -L 1 bash -c "POLICY_PORT=\$2 $PY run_tests.py policy \$0 \$1 2>/dev/null | grep '^{\"cmd\"' >> results/policy.jsonl"
echo "done $(date)"
