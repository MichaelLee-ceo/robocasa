#!/bin/bash
cd "$(dirname "$0")/../.."
PY="$(cd ../.. && pwd)/envs/groot/bin/python -I"
cat recover/inspect/list.txt | CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P 12 -I{} bash -c "$PY inspect_trial.py recover/records/{} 2>/dev/null | grep '^{' > recover/inspect/{}.jsonl"
echo done
