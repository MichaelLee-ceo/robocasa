#!/bin/bash
# Keeps the policy servers of servers.sh alive for jobs that have no watchdog of their own (e.g. ../20261029_closedloop):
# every 2 minutes, any of ports 5751-5758 (GPU0) / 5761-5768 (GPU3) that is neither listening nor starting is restarted.
# Detached; stop it with: pkill -f keep_servers.sh. Log: logs/keep_servers.log.
# usage: setsid nohup ./keep_servers.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")"
declare -A GPU_OF=([5751]=0 [5752]=0 [5753]=0 [5754]=0 [5755]=0 [5756]=0 [5757]=0 [5758]=0 [5761]=3 [5762]=3 [5763]=3 [5764]=3 [5765]=3 [5766]=3 [5767]=3 [5768]=3)
mkdir -p logs
echo "$(date '+%m-%d %H:%M:%S') start (pid $$)" >> logs/keep_servers.log
while true; do
  for P in "${!GPU_OF[@]}"; do
    if ! ss -ltn | grep -q ":$P " && ! pgrep -f "serve_seeded.py .*--port $P\$" > /dev/null; then
      echo "$(date '+%m-%d %H:%M:%S') RESTART server port $P on GPU${GPU_OF[$P]}" >> logs/keep_servers.log
      setsid nohup ./servers.sh ${GPU_OF[$P]} $P 1 >> logs/servers_restart.out 2>&1 < /dev/null &
    fi
  done
  sleep 120
done
