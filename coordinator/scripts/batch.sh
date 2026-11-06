#!/bin/bash
# scripts/run.py point over the phase 0 failure points (not late, MIN_LEFT or more chunks left) x policy seeds, J workers
# on one GPU; skips runs whose <OUT>/<config>/<id>_p<seed>.json exists. SHARD=k/n: every n-th run from the k-th (one
# runner per GPU). The queue is ordered and sharded exactly as runs/20261105_recovery/loop.sh does, so a run lands on
# the same GPU and GR00T server as loop.py's run of the same point and seed (same SHARD/PORTS per GPU). The lines
# must be byte-identical to loop.sh's: sort uses the locale's collation, which skips spaces, so every field orders. VLM="claude:claude-opus-5-5" for the VLM configs (ask before large API spends).
# usage: [PORTS="5581 5582"] [SHARD=k/n] [VLM=spec] setsid nohup coordinator/scripts/batch.sh <gpu> <jobs> <config> "<pseeds>" \
#          > runs/20261106_coordinator/<config>_<gpu>.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
G=$1; J=$2; CONFIG=$3; PSEEDS=${4:-"0 1"}; MIN_LEFT=${MIN_LEFT:-20}
PORTS=(${PORTS:-5581 5582}); SHARD=${SHARD:-0/1}; OUT=${OUT:-runs/20261106_coordinator}
PY="${GROOT_PY:-$(pwd)/envs/groot/bin/python} -I"
VLMARG=${VLM:+--vlm $VLM}
mkdir -p $OUT/$CONFIG
IDS=$(python3 -c "import json; print(' '.join(p['id'] for p in json.load(open('runs/20261105_recovery/phase0/points.json')) if p['kind'] != 'late' and p['chunks_left'] >= $MIN_LEFT))")
i=0
for p in $PSEEDS; do for id in $IDS; do
  [ -f "$OUT/$CONFIG/${id}_p$p.json" ] && continue
  echo "$id retrace $p ${PORTS[$((i % ${#PORTS[@]}))]}"; i=$((i + 1))   # loop.sh's line for its retrace arm, byte for byte
done; done | sort -t_ -k1,1r | awk -v k=${SHARD%/*} -v n=${SHARD#*/} 'NR % n == k' | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$J" -L 1 bash -c "POLICY_PORT=\$3 $PY coordinator/scripts/run.py point \$0 $CONFIG \$2 --out $OUT $VLMARG 2>/dev/null | grep '^{' >> $OUT/$CONFIG.jsonl"
echo "finished $(date)"
