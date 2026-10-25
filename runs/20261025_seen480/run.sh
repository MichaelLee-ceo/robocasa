#!/bin/bash
# Composite-Seen rerun with seeded policy + per-step sim states: one serve_seeded.py server on GPU0 and one on GPU3,
# workers alternate between them (worker w -> GPU0 if w is even, GPU3 if odd) and render on that GPU.
# A rerun of the same out dir skips jobs that already have a result line.
# usage: run.sh <out dir> <jobs.json> <workers per GPU>
set -u
OUT=$1; JOBS=$2; NPG=$3
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
GPUS=(0 3); PORTS=(5591 5593)
mkdir -p "$OUT"; cp "$JOBS" "$OUT/jobs.json"
SRV=()
for i in 0 1; do
  G=${GPUS[$i]}; P=${PORTS[$i]}
  if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; exit 1; fi
  (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G exec python $ROBOCASA_ROOT/runs/scripts/serve_seeded.py \
     --model_path $CK --port $P) > "$OUT/server_gpu$G.log" 2>&1 < /dev/null &
  SRV+=($!)
done
for i in 0 1; do
  P=${PORTS[$i]}
  until ss -ltn | grep -q ":$P "; do kill -0 ${SRV[$i]} 2>/dev/null || { echo "server on port $P died"; exit 1; }; sleep 5; done
done
NW=$((2 * NPG))
echo "$(date +%H:%M:%S) servers up (pids ${SRV[*]}), $NW workers"
cd "$OUT"
WP=()
for w in $(seq 0 $((NW - 1))); do
  i=$((w % 2)); G=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G python -I $ROBOCASA_ROOT/runs/scripts/atomic_eval.py --jobs jobs.json \
    --worker $w --n_workers $NW --out "$OUT" --port ${PORTS[$i]} > worker_$w.log 2>&1 &
  WP+=($!)
done
echo "${WP[*]}" > workers.pid
wait "${WP[@]}"
kill "${SRV[@]}"
echo "$(date +%H:%M:%S) done: $(cat results_w*.jsonl | grep -vc '"error"') results, $(cat results_w*.jsonl | grep -c '"error"') errors"
