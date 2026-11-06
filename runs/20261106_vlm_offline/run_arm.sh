#!/bin/bash
# one model x one context arm over the 90 phase 0 runs: ./run_arm.sh <model dir name> <vlm spec> <arm> <workers> <egl gpu>
# (from the repository root; resumable: runs with an output file are skipped)
cd "$(dirname "$0")/../.."
D=runs/20261106_vlm_offline/$1/$3
mkdir -p $D
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$5 MUJOCO_EGL_DEVICE_ID=$5 MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -a runs/20261106_vlm_offline/runs.txt -P $4 -n 1 sh -c "envs/groot/bin/python -I coordinator/scripts/eval_tracker.py --vlm '$2' --arm $3 --out $D \"\$0\" 2>>$D/stderr.log | grep '^{' >> $D/summary.jsonl"
echo "finished $(date)" >> $D/summary.jsonl.done
