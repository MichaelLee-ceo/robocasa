#!/bin/bash
# Determinism acceptance test (2026-10-25) for serve_seeded.py + atomic_eval.py's policy_seed: the same 2 frozen-scene
# episodes (KettleBoiling s1, StackBowlsCabinet s13; composite_seen checkpoint-60000, baseline_seed, original horizons)
# with a fresh seeded server per run. Compare the per-chunk fingerprints in <run>/results_w*.jsonl ("digest") across runs.
# usage: run_test.sh <run name> <gpu> <port> <n_workers> <policy_seed>
set -u
NAME=$1; GPU=$2; P=$3; NW=$4; PS=$5
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PIP_CONFIG_FILE=/dev/null
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
OUT=$ROBOCASA_ROOT/runs/20261025_determinism/$NAME
mkdir -p "$OUT"
python3 - "$OUT" "$PS" <<'PY'
import json, sys
out, ps = sys.argv[1], int(sys.argv[2]); R = "/home/michaellee/mclee/robocasa/runs/"
jobs = [dict(task=t, seed=s, scene=f"{R}scenes_seen5_v2/{t}_s{s}.npz", condition="baseline_seed", horizon=h,
             tag=f"{t}_s{s}_single", policy_seed=ps)
        for t, s, h in (("KettleBoiling", 1, 1000), ("StackBowlsCabinet", 13, 1400))]
json.dump(jobs, open(f"{out}/jobs.json", "w"), indent=1)
PY
if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; exit 1; fi
(cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && exec nice -n 10 python $ROBOCASA_ROOT/runs/scripts/serve_seeded.py \
   --model_path $CK --port $P) > "$OUT/server.log" 2>&1 < /dev/null &
SRV=$!
until ss -ltn | grep -q ":$P "; do kill -0 $SRV 2>/dev/null || { echo "server died, see $OUT/server.log"; exit 1; }; sleep 5; done
echo "$(date +%H:%M:%S) $NAME: server up on GPU$GPU port $P (pid $SRV), $NW worker(s), policy_seed $PS"
cd "$OUT"
WP=()
for w in $(seq 0 $((NW - 1))); do
  nice -n 10 ionice -c2 -n7 python -I $ROBOCASA_ROOT/runs/scripts/atomic_eval.py --jobs jobs.json --worker $w \
    --n_workers $NW --out "$OUT" --port $P > worker_$w.log 2>&1 &
  WP+=($!)
done
wait "${WP[@]}"
kill $SRV
echo "$(date +%H:%M:%S) $NAME: done"
