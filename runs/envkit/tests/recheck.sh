#!/bin/bash
# Re-run `check` on the scenes listed in results/recheck_list.txt ("<task> <seed>" per line), N jobs on one GPU,
# into results/check_rerun.jsonl (the first campaign rows that predate the metadata filter of the reset comparison).
# usage: setsid nohup ./recheck.sh <gpu> <jobs> > results/recheck.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
GPU=${1:-1}; N=${2:-8}
PY="${GROOT_PY:-$(cd ../../.. && pwd)/envs/groot/bin/python} -I"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$N" -L 1 bash -c "$PY run_tests.py check \$0 \$1 2>/dev/null | grep '^{\"cmd\"' >> results/check_rerun.jsonl" < results/recheck_list.txt
echo "done $(date)"
