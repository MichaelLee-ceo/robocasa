#!/bin/bash
# Start NS seeded policy servers on GPU G (ports PORT0..PORT0+NS-1) and keep them up; pools attach with PORTS="...".
# usage: setsid nohup ./servers.sh <G> <PORT0> [NS=4] > <log> 2>&1 &
G=$1; PORT0=$2; NS=${3:-4}
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
mkdir -p run_v5_queue
for s in $(seq 0 $((NS - 1))); do
  P=$((PORT0 + s))
  if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; continue; fi
  (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 exec python \
     $ROBOCASA_ROOT/runs/scripts/serve_seeded.py --model_path $CK --port $P) > run_v5_queue/server_$P.log 2>&1 < /dev/null &
done
wait
