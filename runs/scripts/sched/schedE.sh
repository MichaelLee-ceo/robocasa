#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ~/Desktop/robocasa/envs/groot
export NO_ALBUMENTATIONS_UPDATE=1
SD=~/Desktop/robocasa/runs/20261012_atomic_seeded; S=~/Desktop/robocasa/runs/scripts/atomic_eval.py
until grep -q "D2 done" /tmp/claude-1001/rc/schedD.log; do sleep 20; done
until ss -ltn | grep -q ":5556 "; do sleep 10; done
echo "$(date +%H:%M) starting E"
cd $SD; for w in $(seq 0 5); do python -I $S --jobs jobs_E.json --worker $w --n_workers 6 --out $SD --port 5556 > worker_E_$w.log 2>&1 & done; wait
echo "$(date +%H:%M) E done"
