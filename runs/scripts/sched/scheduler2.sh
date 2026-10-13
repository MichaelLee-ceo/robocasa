#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ~/Desktop/robocasa/envs/groot
export NO_ALBUMENTATIONS_UPDATE=1
S=~/Desktop/robocasa/runs/scripts/atomic_eval.py
RT=~/Desktop/robocasa/runs/20261012_atomic_repeat_test; SD=~/Desktop/robocasa/runs/20261012_atomic_seeded; A=~/Desktop/robocasa/runs/20261012_atomic_decomp_seen
cd $SD; for w in 0 1 2; do setsid python -I $S --jobs jobs.json --worker $w --n_workers 3 --out $SD --port 5555 > worker_$w.log 2>&1 & done
cd $RT; (for w in 0 1; do python -I $S --jobs jobs.json --worker $w --n_workers 2 --out $RT --port 5555 > worker_$w.log 2>&1 & done; wait
 echo "$(date +%H:%M) repeat test done"; cd $A; for w in 0 1; do python -I $S --jobs jobs_chain_task.json --worker $w --n_workers 2 --out $A --port 5555 > worker_ct_$w.log 2>&1 & done; wait; echo "$(date +%H:%M) chain_task done") &
wait
echo "$(date +%H:%M) all done"
