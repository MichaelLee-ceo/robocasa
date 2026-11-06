#!/bin/bash
# Phase 0: plain.py on TASKS x seeds S0..S1, N workers over the composite servers -> phase0/plain.jsonl
# usage: setsid nohup ./phase0.sh <gpu> <jobs> <seed0> <seed1> "<tasks>" > phase0/run.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
G=${1:-1}; N=${2:-16}; S0=${3:-0}; S1=${4:-29}
TASKS=${5:-"KettleBoiling SteamInMicrowave PackIdenticalLunches"}
PORTS=(5581 5582)
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
mkdir -p phase0
i=0
# longest tasks first, so the tail of the run is short
for t in $TASKS; do for s in $(seq $S0 $S1); do
  [ -f "phase0/records/${t}_s${s}.pkl.gz" ] && continue
  echo "$t $s ${PORTS[$((i % 2))]}"; i=$((i + 1))
done; done | sort -k1,1r | \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  xargs -P "$N" -L 1 bash -c "POLICY_PORT=\$2 $PY plain.py \$0 \$1 2>/dev/null | grep '^{' >> phase0/plain.jsonl"
echo "finished $(date)"
