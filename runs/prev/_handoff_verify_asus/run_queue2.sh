#!/bin/bash
# Detached policy queue (runs past the tool's 2 h limit). Waits for the previous queue (pid $1) to exit, then runs
# batches "<Task> <lo-hi>" in order. Before each batch: wait until every scene file of the batch exists and its
# generator is not still writing that task, then pick the worker count from GPU0 free memory (server 6.5 GB +
# n x 2.8 GB + 1.4 GB if a scene generator runs + 2 GB margin, n <= 6). Episodes already in results_w*.jsonl are
# skipped by atomic_eval (resume by tag). usage: run_queue2.sh <prev_pid> <out_dir> "<Task> <lo-hi>" ...
set -u
PREV=$1; OUT=$2; shift 2
D=$(dirname "$(readlink -f "$0")"); R=/home/michaellee/mclee/robocasa/runs
while kill -0 "$PREV" 2>/dev/null; do sleep 30; done
echo "$(date +%T) previous queue ended"
for b in "$@"; do
  set -- $b; T=$1; S=$2; lo=${S%-*}; hi=${S#*-}
  while true; do
    miss=0
    for s in $(seq $lo $hi); do
      if [ $s -lt 20 ] && [ -f $R/scenes_seen5_v2/${T}_s$s.npz ]; then continue; fi
      [ -f $R/scenes_seen5_v2_s30/${T}_s$s.npz ] || [ -f $R/scenes_seen11_v2/${T}_s$s.npz ] || miss=1
    done
    pgrep -u "$USER" -f "make_scenes.py --tasks $T " > /dev/null && miss=1
    [ $miss -eq 0 ] && break
    sleep 120
  done
  while true; do
    read used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 | tr -d ',')
    gen=0; pgrep -u "$USER" -f make_scenes.py > /dev/null && gen=1
    n=$(python3 -c "f=($total-$used)/1024; n=int((f-6.5-2-1.4*$gen)//2.8); print(max(0,min(6,n)))")
    [ "$n" -ge 1 ] && break
    sleep 120
  done
  echo "$(date '+%F %T') $T s$S: $n worker(s) (GPU0 free $((total-used)) MiB; scene generation running: $gen)" | tee -a "$OUT/NOTE.txt"
  bash "$D/run_seen.sh" "$OUT" "$T" "$S" "$n" 0 5571 >> "$OUT/run.log" 2>&1
  echo "$(date +%T) $T s$S done (exit $?)"
done
echo "$(date +%T) queue2 finished"
