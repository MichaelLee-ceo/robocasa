#!/bin/bash
# Watchdog of a collection (jobs file as argument), 2026-10-29. Runs detached (setsid), so it outlives
# the Claude session that started it. Every 2 minutes it:
#  - restarts any policy server that is down: ports 5751-5754 on GPU0, 5761-5764 on GPU3 (servers.sh, one port);
#  - runs qc.py on the new rows (flags go to logs/flags.jsonl);
#  - logs a STALL if no row was finished for 90 min while workers run.
# When no worker is left:
#  - jobs without an ok row are retried (pool.sh, attached to the servers), at most 2 times;
#  - a job whose error is a live-assert failure (start state, shared first chunk) is a bug, not bad luck: its claim
#    becomes "blocked:assert" so no retry takes it, and it is logged as BLOCKED.
# When nothing is left to retry: DONE, exit.
# Log: logs/<jobs name>/watchdog.log; heartbeat: watchdog.status there; DONE when finished.
# usage: setsid nohup ./watchdog.sh <jobs.json> > logs/<jobs name>.watchdog.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
HERE=$PWD; JOBS=$(realpath "$1"); Q=$HERE/logs/$(basename "$1" .json); mkdir -p $Q
PY="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
WPAT="^python -I $HERE/collect.py --jobs $JOBS"
declare -A GPU_OF=([5751]=0 [5752]=0 [5753]=0 [5754]=0 [5761]=3 [5762]=3 [5763]=3 [5764]=3)
retries=0; last_ok=-1; last_change=$(date +%s)
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $Q/watchdog.log; }
log "watchdog start (pid $$)"

state() {  # prints: <ok> <unfinished> <blocked_now>; blocks assert-failed jobs when asked ($1 = block)
  $PY - "$JOBS" "${1:-}" <<'EOF'
import glob, json, os, socket, sys
jobs, block = json.load(open(sys.argv[1])), sys.argv[2] == "block"
ASSERT = ("start state differs", "replayed first chunk diverged")
ok = unfinished = blocked = 0
cache = {}
for j in jobs:
    out = j["out"]
    if out not in cache:
        rows = {}
        for p in glob.glob(f"{out}/results_w*.jsonl"):
            for l in open(p):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue
                key = (r.get("tag"), r.get("t"))
                prev = rows.get(key)
                if prev is None or "error" in prev:  # an ok row wins over error rows
                    rows[key] = r
        cache[out] = rows
    r = cache[out].get((j["tag"], j["t"]))
    if r is not None and "error" not in r:
        ok += 1
        continue
    unfinished += 1
    if r is not None and any(a in r["error"] for a in ASSERT):
        c = f"{out}/claims/{j['tag']}_t{j['t']}"
        if block and (not os.path.exists(c) or open(c).read() != "blocked:assert"):
            os.makedirs(os.path.dirname(c), exist_ok=True)
            open(c, "w").write("blocked:assert")
            print(f"BLOCKED {out.split('/')[-1]} {j['tag']} t{j['t']}: {r['error'][:200]}", file=sys.stderr)
            blocked += 1
print(ok, unfinished, blocked)
EOF
}

while true; do
  # 1. servers
  for P in "${!GPU_OF[@]}"; do
    if ! ss -ltn | grep -q ":$P " && ! pgrep -f "serve_seeded.py .*--port $P\$" > /dev/null; then
      log "RESTART server port $P on GPU${GPU_OF[$P]} (was down)"
      setsid nohup ./servers.sh ${GPU_OF[$P]} $P 1 >> $Q/servers_restart.out 2>&1 < /dev/null &
    fi
  done
  # 2. quality check of the new rows
  $PY qc.py >> $Q/qc.out 2>&1 || log "qc.py failed (see qc.out)"
  nflags=$(wc -l < $HERE/logs/flags.jsonl 2>/dev/null || echo 0)
  read -r ok unfinished _ <<< "$(state 2>>$Q/watchdog.log)"
  nw=$(pgrep -f -c "$WPAT")
  now=$(date +%s)
  [ "$ok" != "$last_ok" ] && { last_ok=$ok; last_change=$now; }
  echo "$(date '+%m-%d %H:%M:%S') ok $ok / $(($ok + $unfinished)), workers $nw, retries $retries, flags $nflags" > $Q/watchdog.status
  if [ "$nw" -gt 0 ] && [ $((now - last_change)) -gt 5400 ]; then log "STALL: no new row for $(((now - last_change) / 60)) min with $nw workers"; last_change=$now; fi
  # 3. retries once every worker has exited
  if [ "$nw" -eq 0 ]; then
    read -r ok unfinished blocked <<< "$(state block 2>>$Q/watchdog.log)"
    retryable=$((unfinished - blocked))
    # jobs blocked in an earlier pass still count as unfinished; count the retryable ones directly
    retryable=$($PY - "$JOBS" <<'EOF'
import glob, json, os, sys
n = 0
for j in json.load(open(sys.argv[1])):
    out = j["out"]; c = f"{out}/claims/{j['tag']}_t{j['t']}"
    if os.path.exists(c) and open(c).read() == "blocked:assert":
        continue
    ok = any("error" not in json.loads(l) for p in glob.glob(f"{out}/results_w*.jsonl") for l in open(p)
             if f'"{j["tag"]}"' in l and json.loads(l).get("t") == j["t"])
    n += not ok
print(n)
EOF
)
    if [ "$retryable" -gt 0 ] && [ $retries -lt 2 ]; then
      retries=$((retries + 1)); n=$(( (retryable + 1) / 2 )); [ $n -gt 12 ] && n=12
      log "RETRY $retries: $retryable unfinished jobs, $n workers each on GPU0 and GPU3 ($ok ok)"
      G=0 PORTS="5751 5752 5753 5754" W0=$((700 + retries * 40)) setsid nohup ./pool.sh $Q $JOBS $n > $Q/pool_retry${retries}_gpu0.out 2>&1 < /dev/null &
      G=3 PORTS="5761 5762 5763 5764" W0=$((720 + retries * 40)) setsid nohup ./pool.sh $Q $JOBS $n > $Q/pool_retry${retries}_gpu3.out 2>&1 < /dev/null &
      sleep 120; continue
    fi
    log "collection finished: $ok ok, $retryable unfinished after $retries retries"
    $PY qc.py >> $Q/qc.out 2>&1
    log "DONE"; touch $Q/DONE; exit 0
  fi
  sleep 120
done
