#!/bin/bash
ROBOCASA_ROOT="${ROBOCASA_ROOT:-$HOME/Desktop/robocasa}"; RC_DIR="${RC_DIR:-/tmp/claude-1001/rc}"; export ROBOCASA_ROOT RC_DIR
mkdir -p "$RC_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
R=$ROBOCASA_ROOT/runs/20261012_groot_repro_seen_h1x; A=$ROBOCASA_ROOT/runs/20261012_atomic_decomp_seen
export NO_ALBUMENTATIONS_UPDATE=1
while [ $(ps -eo args | grep -c "^python -I .*groot_eval.py --jobs jobs.json --worker [012] --n_workers 5") -gt 0 ]; do sleep 30; done
echo "$(date +%H:%M) repro workers 0-2 done; starting repro 3,4 and atomic x3"
cd $R; for w in 3 4; do setsid python -I $ROBOCASA_ROOT/runs/scripts/groot_eval.py --jobs jobs.json --worker $w --n_workers 5 --out $R --port 5555 >> worker_$w.log 2>&1 & done
kill $(ps -eo pid,args | grep "^ *[0-9]* python -I .*atomic_eval.py --jobs jobs.json --worker 0 --n_workers 1" | awk '{print $1}') 2>/dev/null
sleep 10
cd $A; for w in 0 1 2; do setsid python -I $ROBOCASA_ROOT/runs/scripts/atomic_eval.py --jobs jobs.json --worker $w --n_workers 3 --out $A --port 5555 >> worker_$w.log 2>&1 & done
wait
echo "$(date +%H:%M) all done"
