#!/bin/bash
# Stage-level imagine-then-choose experiment on GPU1 (approved by the user on 2026-10-26: GPU0/GPU3 were taken by
# another user's training). NS seeded policy servers + NW workers, all on GPU1.
# usage: run.sh <out dir> <jobs.json> <n workers> [n servers=4]
set -u
OUT=$(realpath -m "$1"); JOBS=$(realpath "$2"); NW=$3; NS=${4:-4}
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; cp "$JOBS" "$OUT/jobs.json"
G=1; PORTS=(); SRV=()
for s in $(seq 0 $((NS - 1))); do
  P=$((5711 + s))
  if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; exit 1; fi
  (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 exec python \
     $ROBOCASA_ROOT/runs/scripts/serve_seeded.py --model_path $CK --port $P) > "$OUT/server_$P.log" 2>&1 < /dev/null &
  SRV+=($!); PORTS+=($P)
done
echo "${SRV[*]}" > "$OUT/servers.pid"
for i in "${!PORTS[@]}"; do
  until ss -ltn | grep -q ":${PORTS[$i]} "; do kill -0 ${SRV[$i]} 2>/dev/null || { echo "server ${PORTS[$i]} died"; exit 1; }; sleep 5; done
done
echo "$(date +%H:%M:%S) $NS servers up"
cd "$OUT"; WP=()
for w in $(seq 0 $((NW - 1))); do
  P=${PORTS[$((w % NS))]}
  CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I $HERE/imagine.py \
    --jobs jobs.json --worker $w --n_workers $NW --out "$OUT" --port $P >> worker_$w.log 2>&1 &
  WP+=($!)
done
echo "${WP[*]}" > workers.pid
wait "${WP[@]}"
kill "${SRV[@]}"
echo "$(date +%H:%M:%S) done: $(cat results_w*.jsonl | grep -vc '"error"') results, $(cat results_w*.jsonl | grep -c '"error"') errors"
