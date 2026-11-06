#!/bin/bash
# Every instance x recovery x policy seed of recover.py, N jobs on GPU G -> recover/results.jsonl
# usage: setsid nohup ./recover.sh <gpu> <jobs> > recover/run.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
G=${1:-1}; N=${2:-8}
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
mkdir -p recover
IDS="DS1_short DS2_wrong DS0_wander GTB0_short GTB1_short GTB2_short KB2_air SUCS0_drop"
for p in 0 1; do for a in continue resample stage_instr atomic backoff; do for i in $IDS; do echo "$i $a $p"; done; done; done | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$N" -L 1 bash -c "$PY recover.py \$0 \$1 \$2 2>/dev/null | grep '^{' >> recover/results.jsonl"
echo "done $(date)"
