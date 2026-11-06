#!/bin/bash
# loop.py over the phase 0 failure points (not late, MIN_LEFT or more chunks left) x ARMS x policy seeds, J workers
# usage: [PORTS="5581 5582"] [SHARD=k/n] setsid nohup ./loop.sh <gpu> <jobs> "<arms>" "<pseeds>" > loop/run.out 2>&1 < /dev/null &
# (SHARD=k/n: every n-th run from the k-th, so n runners on n GPUs split one queue; .claim files stop any overlap)
# (each run also writes loop/<arm>/<id>_p<seed>.json, which loop_score.py reads)
cd "$(dirname "$0")"
G=${1:-1}; J=${2:-16}; ARMS=${3:-"control full"}; PSEEDS=${4:-"0 1"}; MIN_LEFT=${MIN_LEFT:-20}
PORTS=(${PORTS:-5581 5582}); SHARD=${SHARD:-0/1}
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
mkdir -p loop
IDS=$(python3 -c "import json; print(' '.join(p['id'] for p in json.load(open('phase0/points.json')) if p['kind'] != 'late' and p['chunks_left'] >= $MIN_LEFT))")
i=0
for p in $PSEEDS; do for a in $ARMS; do for id in $IDS; do
  [ -f "loop/$a/${id}_p$p.json" ] && continue
  echo "$id $a $p ${PORTS[$((i % ${#PORTS[@]}))]}"; i=$((i + 1))
done; done; done | sort -t_ -k1,1r | awk -v k=${SHARD%/*} -v n=${SHARD#*/} 'NR % n == k' | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$J" -L 1 bash -c "POLICY_PORT=\$3 $PY loop.py \$0 \$1 \$2 2>/dev/null | grep '^{' >> loop/results.jsonl"
echo "finished $(date)"
