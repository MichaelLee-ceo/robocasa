#!/bin/bash
# Run batches "<Task> <seeds>" one after another on GPU0 with run_seen.sh. Before each batch the worker count is
# recomputed from GPU0's free memory: server 6.5 GB + n x 2.8 GB + 1.4 GB if a scene generator is running + 2 GB
# margin must fit; n is capped at 6. If not even 1 worker fits, wait and re-check. Each batch is logged in NOTE.txt.
# usage: run_queue.sh <out_dir> "<Task> <lo-hi>" ["<Task> <lo-hi>" ...]
set -u
OUT=$1; shift
D=$(dirname "$(readlink -f "$0")")
for b in "$@"; do
  set -- $b; T=$1; S=$2
  while true; do
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 | tr -d ',')
    gen=0; pgrep -u "$USER" -f make_scenes.py > /dev/null && gen=1
    n=$(python3 -c "f=($total-$used)/1024; n=int((f-6.5-2-1.4*$gen)//2.8); print(max(0,min(6,n)))")
    [ "$n" -ge 1 ] && break
    echo "$(date +%T) $T $S: not enough GPU0 memory (free $(( (total-used)/1024 )) GB, scene gen $gen), waiting"; sleep 120
  done
  echo "$(date '+%F %T') $T s$S: $n worker(s) (GPU0 free $(( (total-used) )) MiB; scene generation running: $gen)" | tee -a "$OUT/NOTE.txt"
  bash "$D/run_seen.sh" "$OUT" "$T" "$S" "$n" 0 5571 >> "$OUT/run.log" 2>&1
  echo "$(date +%T) $T s$S done (exit $?)"
done
