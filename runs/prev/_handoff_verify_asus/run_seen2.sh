#!/bin/bash
# run_seen2.sh = run_seen.sh + STOP file + reuse check of an existing server. STOP: when $RC_DIR/STOP exists, each
# worker is stopped by PID right after it writes its next result line (or after 40 min), no new pass starts, and the
# server is stopped if this script started it. Same as run_accept.sh, but every task gets its original horizon (runs/20261012_groot_repro_seen_h1x/horizons_original.json).
# README §5.3 policy check on ASUS-GPU (same settings as runs/20261013_accept: composite_seen checkpoint-60000,
# baseline_seed on the frozen v2 scenes, horizon 1400). Unlike runs/scripts/sched/*.sh this never kills other
# processes: it starts its own policy server (if the port is free), and stops only that server at the end.
# usage: run_accept.sh <out_dir> <tasks comma-separated> <seeds e.g. 0-19> <n_workers> [gpu=0] [port=5571]
set -u
OUT=$1; TASKS=$2; SEEDS=$3; NW=$4; GPU=${5:-0}; P=${6:-5571}
source ~/miniconda3/etc/profile.d/conda.sh && conda activate /home/michaellee/mclee/robocasa/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PIP_CONFIG_FILE=/dev/null
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
R=$ROBOCASA_ROOT/runs; CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
mkdir -p "$OUT" "$RC_DIR"
python3 - "$OUT" "$TASKS" "$SEEDS" <<'PY'
import json, os, sys
out, tasks, seeds = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
lo, hi = map(int, seeds.split("-")) if "-" in seeds else (int(seeds), int(seeds))
R_ = os.environ["ROBOCASA_ROOT"] + "/runs/"
# s0-s19 of the 5 original tasks keep their scenes_seen5_v2 files; everything else comes from the 30-seed folders
def SC_(t, s):
    for d in ("scenes_seen5_v2", "scenes_seen5_v2_s30", "scenes_seen11_v2"):
        if os.path.exists(f"{R_}{d}/{t}_s{s}.npz") and (d != "scenes_seen5_v2" or s < 20):
            return f"{R_}{d}/{t}_s{s}.npz"
    raise SystemExit(f"no frozen scene for {t} s{s}")
H = json.load(open(os.environ["ROBOCASA_ROOT"] + "/runs/20261012_groot_repro_seen_h1x/horizons_original.json"))
json.dump([dict(task=t, seed=s, scene=SC_(t, s), condition="baseline_seed", horizon=H[t], tag=f"{t}_s{s}_single")
           for t in tasks for s in range(lo, hi + 1)], open(f"{out}/jobs.json", "w"))
PY
[ $? -eq 0 ] || { echo "job list failed (missing scene?)"; exit 1; }
SRV=""
if ss -ltn | grep -q ":$P "; then  # reuse an existing server only if it is ours and serves composite_seen
  EP=$(ss -ltnp 2>/dev/null | grep ":$P " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  if [ -z "$EP" ] || [ "$(ps -o user= -p $EP)" != "$USER" ] || ! ps -o args= -p $EP | grep -q 'composite_seen/checkpoint-60000'; then
    echo "port $P is held by pid ${EP:-?} that is not our composite_seen server; not starting"; exit 1
  fi
  echo $EP > "$RC_DIR/server_$P.pid"
else
  (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && exec python scripts/run_eval.py --server --model_path $CK \
     --task_set composite_seen --split target --port $P) > "$RC_DIR/server_asus_$P.log" 2>&1 < /dev/null &
  SRV=$!
  echo $SRV > "$RC_DIR/server_$P.pid"
  until ss -ltn | grep -q ":$P "; do kill -0 $SRV 2>/dev/null || { echo "server died, see $RC_DIR/server_asus_$P.log"; exit 1; }; sleep 5; done
fi
echo "$(date +%H:%M:%S) server up on port $P (pid ${SRV:-existing}), $NW workers"
cd "$OUT"
STOPF="$RC_DIR/STOP"
for pass in 1 2; do  # pass 2 retries jobs that ended with an error (same as runjobs in sched/*.sh)
  [ -f "$STOPF" ] && break
  declare -A WPID=() N0=()
  for w in $(seq 0 $((NW - 1))); do
    python -I $R/scripts/atomic_eval.py --jobs jobs.json --worker $w --n_workers $NW --out "$OUT" --port $P >> worker_$w.log 2>&1 &
    WPID[$w]=$!
  done
  T0=""
  while true; do
    alive=0; for w in "${!WPID[@]}"; do kill -0 ${WPID[$w]} 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break
    if [ -f "$STOPF" ]; then
      if [ -z "$T0" ]; then T0=$(date +%s); for w in "${!WPID[@]}"; do N0[$w]=$(cat results_w$w.jsonl 2>/dev/null | wc -l); done; echo "$(date +%T) STOP seen, stopping workers after their current episode"; fi
      for w in "${!WPID[@]}"; do
        n=$(cat results_w$w.jsonl 2>/dev/null | wc -l)
        if kill -0 ${WPID[$w]} 2>/dev/null && { [ $n -gt ${N0[$w]} ] || [ $(( $(date +%s) - T0 )) -gt 2400 ]; }; then kill ${WPID[$w]}; fi
      done
    fi
    sleep 10
  done
  unset WPID N0
done
echo "$(date +%H:%M:%S) workers done"
[ -n "$SRV" ] && kill $SRV
python3 - "$OUT" <<'PY'
import glob, json, sys, collections
R = [json.loads(l) for p in glob.glob(sys.argv[1] + "/results_w*.jsonl") for l in open(p)]
c = collections.Counter((r["task"], "error" if "error" in r else r["success"]) for r in R)
print(dict(c))
PY
