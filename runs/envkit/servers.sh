#!/bin/bash
# Start NS seeded GR00T policy servers (serve_seeded.py, checkpoint-60000) on GPU G, ports PORT0..PORT0+NS-1.
# usage: setsid nohup ./servers.sh <G> <PORT0> [NS=2] > /dev/null 2>&1 < /dev/null &   (logs: logs/server_<port>.log)
G=$1; PORT0=$2; NS=${3:-2}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
PY="${GROOT_PY:-$ROOT/envs/groot/bin/python}"
CK=$ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
export CUDA_DEVICE_ORDER=PCI_BUS_ID PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROOT/.hf_cache/hub
mkdir -p "$HERE/logs"
for s in $(seq 0 $((NS - 1))); do
  P=$((PORT0 + s))
  if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; continue; fi
  (cd $ROOT/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 exec $PY \
     $HERE/serve_seeded.py --model_path $CK --port $P) > "$HERE/logs/server_$P.log" 2>&1 < /dev/null &
done
wait
