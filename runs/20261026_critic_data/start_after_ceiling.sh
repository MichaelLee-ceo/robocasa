#!/bin/bash
# Start the collection on GPU1 once the ceiling run (../20261026_imagine, multi) has finished and freed GPU1.
# GPU2 was tried first (run_aborted_oom/, then 2 decision points in run/) but its 24 GB peaked at the limit.
cd "$(dirname "$0")"
until grep -q "done:" ../20261026_imagine/multi_resume.out; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 4000 ]; do sleep 30; done
echo "$(date +%H:%M:%S) GPU1 free, starting"
G=1 ./run.sh run jobs.json 14 4
