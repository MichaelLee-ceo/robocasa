#!/bin/bash
# Re-run the policy test for "<task> <seed> <port>" lines of results/policy_rerun_list.txt once policy_tests.sh is done
# (runs made before the latched-success rule of stages.track) -> results/policy_rerun.jsonl (these rows supersede).
cd "$(dirname "$0")"
GPU=${1:-1}; N=${2:-6}
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
while pgrep -f "^/bin/bash ./policy_tests.sh" > /dev/null; do sleep 60; done
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$N" -L 1 bash -c "POLICY_PORT=\$2 $PY run_tests.py policy \$0 \$1 2>/dev/null | grep '^{\"cmd\"' >> results/policy_rerun.jsonl" < results/policy_rerun_list.txt
echo "done $(date)"
