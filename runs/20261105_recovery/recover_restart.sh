#!/bin/bash
# The restart arm of recover.py (fresh attempts from the episode start) for the stage-0 instances, policy seeds 0-3
# usage: setsid nohup ./recover_restart.sh <gpu> <jobs> > recover/restart.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
G=${1:-1}; N=${2:-8}
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
IDS="DS0_wander DS1_short DS2_wrong GTB0_short GTB1_short GTB2_short KB2_air"
for p in 0 1 2 3; do for i in $IDS; do echo "$i restart $p"; done; done | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$N" -L 1 bash -c "$PY recover.py \$0 \$1 \$2 2>/dev/null | grep '^{' >> recover/results.jsonl"
echo "done $(date)"
