#!/bin/bash
# Imagine workers on policy servers that are already running (../20261026_critic_data/servers.sh on GPU0 / GPU3, kept up
# by keep_servers.sh); run.sh instead starts its own servers on GPU1. Two passes: imagine.py skips finished jobs, so the
# second pass retries jobs that ended with an error row. Detached; log: <out>/run.log.
# usage: setsid nohup ./run_on_servers.sh <out dir> <jobs.json> <n workers> <port> [<port> ...] > /dev/null 2>&1 &
set -u
OUT=$(realpath -m "$1"); JOBS=$(realpath "$2"); NW=$3; shift 3; PORTS=("$@")
export ROBOCASA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1
PY="$ROBOCASA_ROOT/envs/groot/bin/python -I"
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
if [ -f "$OUT/jobs.json" ] && ! cmp -s "$JOBS" "$OUT/jobs.json"; then echo "$OUT/jobs.json differs from $JOBS" >&2; exit 1; fi
cp "$JOBS" "$OUT/jobs.json"
cd "$OUT"
for pass in 1 2; do
  echo "$(date '+%m-%d %H:%M:%S') pass $pass: $NW workers on ports ${PORTS[*]}" >> run.log
  WP=()
  for w in $(seq 0 $((NW - 1))); do
    P=${PORTS[$((w % ${#PORTS[@]}))]}; G=$([ "$P" -lt 5760 ] && echo 0 || echo 3)
    CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G $PY "$HERE/imagine.py" --jobs jobs.json --worker $w --n_workers $NW \
      --out "$OUT" --port $P >> worker_$w.log 2>&1 &
    WP+=($!)
  done
  wait "${WP[@]}"
  echo "$(date '+%m-%d %H:%M:%S') pass $pass done: $(cat results_w*.jsonl | grep -vc '"error"') results, $(cat results_w*.jsonl | grep -c '"error"') error rows" >> run.log
done
echo "$(date '+%m-%d %H:%M:%S') all done" >> run.log
