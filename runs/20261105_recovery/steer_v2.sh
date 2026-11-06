#!/bin/bash
# steer v2: rank the candidates from predicted chunks (steer_rank.py), then roll every candidate out
# (steer_probe.py with STEER_DIR=steer/v2, 2 candidates per process, J processes)
# usage: setsid nohup ./steer_v2.sh <gpu> [seeds=5] [chunks=20] [jobs=16] > steer/v2/run.out 2>&1 < /dev/null &
cd "$(dirname "$0")"
G=${1:-1}; S=${2:-5}; C=${3:-20}; J=${4:-16}
PY="${GROOT_PY:-$(cd ../.. && pwd)/envs/groot/bin/python} -I"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
IDS="GTB1_short GTB2_short KB2_air SUCS0_drop DS1_short DS2_wrong"
mkdir -p steer/v2/logs
echo $IDS | tr ' ' '\n' | xargs -P 6 -I{} bash -c "$PY steer_rank.py {} 8 > steer/v2/logs/rank_{}.log 2>&1"
echo "ranked $(date)"
for i in $IDS; do
  n=$(python3 -c "import json; print(len(json.load(open('steer/v2/instructions.json'))['$i']))")
  for ((k=0; k<n; k+=2)); do ks=$k; [ $((k+1)) -lt $n ] && ks="$k,$((k+1))"; echo "$i $ks"; done
done | STEER_DIR=steer/v2 xargs -P "$J" -L 1 bash -c "STEER_KS=\$1 $PY steer_probe.py \$0 $S $C > steer/v2/logs/probe_\$0_\$1.log 2>&1"
echo "finished $(date)"
