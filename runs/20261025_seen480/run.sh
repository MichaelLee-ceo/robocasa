#!/bin/bash
# Composite-Seen rerun with seeded policy + per-step sim states. SPG serve_seeded.py servers on each of GPU0 and GPU3;
# worker w talks to server (w mod #servers) and renders on that server's GPU. A ZeroMQ REP server answers one request
# at a time and its CPU preprocessing dominates, so several servers per GPU with few threads each keep the queue
# short (one server with default threads per GPU ran at ~580% CPU / 0% GPU and stalled 16 workers on 2026-10-25).
# A rerun of the same out dir skips jobs that already have a result line.
# usage: run.sh <out dir> <jobs.json> <workers per GPU> [servers per GPU=4] [threads per server=4]
set -u
OUT=$1; JOBS=$2; NPG=$3; SPG=${4:-4}; ST=${5:-4}
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
mkdir -p "$OUT"; cp "$JOBS" "$OUT/jobs.json"
SGPU=(); SPORT=(); SRV=()
for G in 0 3; do
  for s in $(seq 0 $((SPG - 1))); do
    P=$((5591 + 10 * G + s))
    if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; exit 1; fi
    (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=$ST MKL_NUM_THREADS=$ST exec python \
       $ROBOCASA_ROOT/runs/scripts/serve_seeded.py --model_path $CK --port $P) > "$OUT/server_gpu${G}_$P.log" 2>&1 < /dev/null &
    SRV+=($!); SGPU+=($G); SPORT+=($P)
  done
done
echo "${SRV[*]}" > "$OUT/servers.pid"
for i in "${!SPORT[@]}"; do
  until ss -ltn | grep -q ":${SPORT[$i]} "; do kill -0 ${SRV[$i]} 2>/dev/null || { echo "server on port ${SPORT[$i]} died"; exit 1; }; sleep 5; done
done
NS=${#SPORT[@]}; NW=$((2 * NPG))
echo "$(date +%H:%M:%S) $NS servers up (ports ${SPORT[*]}), $NW workers"
cd "$OUT"
WP=()
for w in $(seq 0 $((NW - 1))); do
  i=$((w % NS)); G=${SGPU[$i]}
  CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I $ROBOCASA_ROOT/runs/scripts/atomic_eval.py \
    --jobs jobs.json --worker $w --n_workers $NW --out "$OUT" --port ${SPORT[$i]} >> worker_$w.log 2>&1 &
  WP+=($!)
done
echo "${WP[*]}" > workers.pid
wait "${WP[@]}"
kill "${SRV[@]}"
echo "$(date +%H:%M:%S) done: $(cat results_w*.jsonl | grep -vc '"error"') results, $(cat results_w*.jsonl | grep -c '"error"') errors"
