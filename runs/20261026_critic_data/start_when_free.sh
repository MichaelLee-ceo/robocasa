#!/bin/bash
# Start a task's collection on GPU0 or GPU3 as soon as one is free (under 2 GB used; another user's training holds both),
# checked every 10 s, first come first served -- but only as many workers as the CPUs allow: workers = min(12, idle - 4),
# idle = nproc - 1-minute load average, and it keeps waiting while fewer than 6 workers would fit.
# usage: setsid nohup ./start_when_free.sh [task=PackIdenticalLunches] >> run_<task>.out 2>&1 &
cd "$(dirname "$0")"
TASK=${1:-PackIdenticalLunches}
NCPU=$(nproc); last=0
echo "$(date +%H:%M:%S) waiting for GPU0 or GPU3, and CPU room, to start $TASK"
while true; do
  for g in 0 3; do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$mem" -lt 2000 ] || continue
    load=$(cut -d" " -f1 /proc/loadavg)
    nw=$(python3 -c "print(max(0, min(12, int($NCPU - $load) - 4)))")
    if [ "$nw" -ge 6 ]; then
      echo "$(date +%H:%M:%S) GPU$g free ($mem MiB), load $load -> $nw workers, starting $TASK"
      G=$g PORT0=5731 exec ./run.sh run_$TASK jobs_$TASK.json $nw 3
    fi
    now=$(date +%s)
    if [ $((now - last)) -ge 600 ]; then echo "$(date +%H:%M:%S) GPU$g free, but load $load leaves room for $nw workers; waiting"; last=$now; fi
  done
  sleep 10
done
