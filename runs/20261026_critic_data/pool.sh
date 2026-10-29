#!/bin/bash
# Shared-queue launcher (collect.py --claim): NW workers on GPU G that take jobs from one queue, so pools on several
# GPUs can fill the same out dir without doing a point twice. Starts NS seeded servers on ports from PORT0, unless PORTS
# lists servers that are already up (those are left running at the end).
# Before starting the workers, it removes claims whose process is gone (a crashed worker or a stopped run), so those
# points are redone; also in the dirs that jobs name in their "out" field. Workers write results_w<W0 + w>.jsonl; use a W0 that no other run on this out dir uses.
# usage: [G=1] [PORT0=5741 | PORTS="5731 5732"] [W0=100] pool.sh <out dir> <jobs.json> <n workers> [n servers=4]
set -u
OUT=$(realpath -m "$1"); JOBS=$(realpath "$2"); NW=$3; NS=${4:-4}
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
HERE=$(cd "$(dirname "$0")" && pwd)
G=${G:-1}; W0=${W0:-100}
mkdir -p "$OUT/claims"
# the out dir, and every dir named by a job's "out" field
for D in "$OUT" $(python3 -c "import json,sys; print(' '.join(sorted({j['out'] for j in json.load(open(sys.argv[1])) if 'out' in j})))" "$JOBS"); do
  for c in "$D"/claims/*; do
    [ -e "$c" ] || continue
    IFS=: read -r h p < "$c"
    if [ "$h" = "$(hostname)" ] && [ -n "$p" ] && [ ! -d /proc/$p ]; then rm -f "$c"; fi
  done
done
SRV=()
if [ -n "${PORTS:-}" ]; then
  read -r -a PORTS <<< "$PORTS"
else
  PORTS=()
  for s in $(seq 0 $((NS - 1))); do
    P=$((${PORT0:-5741} + s))
    if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; exit 1; fi
    (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 exec python \
       $ROBOCASA_ROOT/runs/scripts/serve_seeded.py --model_path $CK --port $P) > "$OUT/server_$P.log" 2>&1 < /dev/null &
    SRV+=($!); PORTS+=($P)
  done
  for i in "${!PORTS[@]}"; do
    until ss -ltn | grep -q ":${PORTS[$i]} "; do kill -0 ${SRV[$i]} 2>/dev/null || { echo "server ${PORTS[$i]} died"; kill "${SRV[@]}"; exit 1; }; sleep 5; done
  done
  echo "$(date +%H:%M:%S) $NS servers up on GPU$G (ports ${PORTS[*]})"
fi
cd "$OUT"; WP=()
for w in $(seq 0 $((NW - 1))); do
  P=${PORTS[$((w % ${#PORTS[@]}))]}
  CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I $HERE/collect.py \
    --jobs "$JOBS" --worker $((W0 + w)) --n_workers $NW --out "$OUT" --port $P --claim >> worker_$((W0 + w)).log 2>&1 &
  WP+=($!)
done
echo "$(date +%H:%M:%S) GPU$G: $NW workers (W$W0..) on ports ${PORTS[*]}"
wait "${WP[@]}"
[ ${#SRV[@]} -gt 0 ] && kill "${SRV[@]}"
echo "$(date +%H:%M:%S) GPU$G pool W$W0 done: $(cat results_w*.jsonl | grep -vc '"error"') results in $OUT, $(cat results_w*.jsonl | grep -c '"error"') error rows"
