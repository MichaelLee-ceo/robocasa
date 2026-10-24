#!/bin/bash
ROBOCASA_ROOT="${ROBOCASA_ROOT:-$HOME/Desktop/robocasa}"; RC_DIR="${RC_DIR:-/tmp/claude-1001/rc}"; export ROBOCASA_ROOT RC_DIR
mkdir -p "$RC_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export NO_ALBUMENTATIONS_UPDATE=1
SD=$ROBOCASA_ROOT/runs/20261012_atomic_seeded; S=$ROBOCASA_ROOT/runs/scripts/atomic_eval.py
until grep -q "D2 done" $RC_DIR/schedD.log; do sleep 20; done
until ss -ltn | grep -q ":5556 "; do sleep 10; done
echo "$(date +%H:%M) starting E"
cd $SD; for w in $(seq 0 5); do python -I $S --jobs jobs_E.json --worker $w --n_workers 6 --out $SD --port 5556 > worker_E_$w.log 2>&1 & done; wait
echo "$(date +%H:%M) E done"
