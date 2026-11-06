#!/bin/bash
# stage_start.py over the loop's failure points (not late, MIN_LEFT or more chunks left) x policy seeds, J workers
# usage: [PORTS="5601 5602"] setsid nohup ./stage_start.sh <gpu> <jobs> "<pseeds>" > stage_start/run_<gpu>.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
G=${1:-1}; J=${2:-12}; PSEEDS=${3:-"0 1"}; MIN_LEFT=${MIN_LEFT:-20}
PORTS=(${PORTS:-5581 5582})
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
mkdir -p stage_start
IDS=$(python3 -c "import json; print(' '.join(p['id'] for p in json.load(open('phase0/points.json')) if p['kind'] != 'late' and p['chunks_left'] >= $MIN_LEFT))")
i=0
for p in $PSEEDS; do for id in $IDS; do
  [ -f "stage_start/${id}_p$p.json" ] && continue
  echo "$id $p ${PORTS[$((i % ${#PORTS[@]}))]}"; i=$((i + 1))
done; done | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$J" -L 1 bash -c "POLICY_PORT=\$2 $PY stage_start.py \$0 \$1 2>/dev/null | grep '^{' >> stage_start/results.jsonl"
echo "finished $(date)"
