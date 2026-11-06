#!/bin/bash
# steer_probe.py for every failure point (8 in parallel), after steer/instructions.json exists
# usage: setsid nohup ./steer_probe.sh <gpu> [seeds] [chunks] > steer/probe.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
G=${1:-1}; S=${2:-4}; C=${3:-8}
until [ -f steer/instructions.json ]; do sleep 10; done
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
echo "DS1_short DS2_wrong DS0_wander GTB0_short GTB1_short GTB2_short KB2_air SUCS0_drop" | tr ' ' '\n' | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P 8 -I{} bash -c "$PY steer_probe.py {} $S $C > steer/probe_{}.log 2>&1; tail -1 steer/probe_{}.log"
echo "done $(date)"
