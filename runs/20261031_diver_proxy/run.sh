#!/bin/bash
# Features at every confirm / control / dense / coarse point, one extract.py worker per feature server (ports 5791-5794,
# started with server.sh). Detached; points already written are skipped, so a rerun resumes.
# usage: setsid nohup ./run.sh > logs/run.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
export ROBOCASA_ROOT="$(cd ../.. && pwd)"
PY=$ROBOCASA_ROOT/envs/groot/bin/python
P=(5791 5792 5793 5794)
for w in 0 1 2 3; do
  $PY -I extract.py --port ${P[$w]} --sets confirm,control,dense,coarse --worker $w --n_workers 4 > logs/extract_w$w.out 2>&1 &
done
wait
echo "$(date '+%m-%d %H:%M:%S') done"
