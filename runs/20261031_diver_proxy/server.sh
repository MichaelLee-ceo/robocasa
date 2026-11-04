#!/bin/bash
# Start one serve_features.py on GPU G, port P (detached by the caller with setsid).
# usage: setsid nohup ./server.sh <G> <P> > logs/server_<P>.log 2>&1 < /dev/null &
G=$1; P=$2
export ROBOCASA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 exec python \
  $ROBOCASA_ROOT/runs/scripts/serve_features.py --model_path $CK --port $P
