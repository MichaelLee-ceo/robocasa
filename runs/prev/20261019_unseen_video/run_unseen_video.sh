#!/bin/bash
# Composite-Unseen rerun WITH videos: same 320 seeded jobs as runs/20261012_groot_repro_h1x (16 tasks x seeds 0-19,
# original horizons), same client (scripts/groot_eval.py, saves mp4 + sim states), official composite_unseen
# checkpoint-60000. One policy server per GPU; workers of each GPU take every other job of a shared list and skip done tags.
# usage: run_unseen_video.sh <gpu> <port> <n_workers> <worker_offset> <total_workers>
set -u
GPU=$1; P=$2; NW=$3; OFF=$4; TOT=$5
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa; OUT=$ROBOCASA_ROOT/runs/20261019_unseen_video
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU PIP_CONFIG_FILE=/dev/null
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_unseen/checkpoint-60000
U=$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$U" -lt 2048 ] || { echo "GPU$GPU busy (${U} MiB used), not starting"; exit 1; }
ss -ltn | grep -q ":$P " && { echo "port $P in use, not starting"; exit 1; }
(cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && exec nice -n 10 python scripts/run_eval.py --server --model_path $CK \
   --task_set composite_unseen --split target --port $P) > $OUT/server_gpu$GPU.log 2>&1 < /dev/null &
SRV=$!; echo $SRV > $OUT/server_gpu$GPU.pid
until ss -ltn | grep -q ":$P "; do kill -0 $SRV 2>/dev/null || { echo "server died, see server_gpu$GPU.log"; exit 1; }; sleep 5; done
echo "$(date +%T) GPU$GPU server up pid $SRV"
cd $OUT; WP=""
for i in $(seq 0 $((NW - 1))); do
  w=$((OFF + i))
  nice -n 10 python -I $ROBOCASA_ROOT/runs/scripts/groot_eval.py --jobs jobs.json --worker $w --n_workers $TOT \
    --out $OUT --port $P >> worker_$w.log 2>&1 &
  WP="$WP $!"
done
echo "workers:$WP" > workers_gpu$GPU.pid
wait $WP; echo "$(date +%T) GPU$GPU workers done"; kill $SRV
