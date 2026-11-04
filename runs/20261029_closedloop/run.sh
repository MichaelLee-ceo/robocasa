#!/bin/bash
# Closed-loop oracle steering: one process per episode on the policy servers of ../20261026_critic_data/servers.sh
# (ports 5751-5758 on GPU0, 5761-5768 on GPU3, kept up by keep_servers.sh), detached; a failed episode is retried once.
# Results: out/<tag>.json; logs: logs/<tag>.log; run.log. PORT_OFFSET shifts the port assignment (to spread several
# launches over the servers).
# usage: setsid nohup ./run.sh <task> <scene> [<scene> ...] > run.out 2>&1 < /dev/null &
#   KettleBoiling pilot (2026-10-29): ./run.sh KettleBoiling 2 4 5 8 9 10 12 17 19 24
cd "$(dirname "$0")"
export ROBOCASA_ROOT="$(cd ../.. && pwd)" CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1
PY="${GROOT_PY:-$ROBOCASA_ROOT/envs/groot/bin/python} -I"
TASK=$1; shift
PORTS=(5751 5761 5752 5762 5753 5763 5754 5764 5755 5765 5756 5766 5757 5767 5758 5768); OFF=${PORT_OFFSET:-0}
mkdir -p out logs
n=0
for s in "$@"; do
  tag=${TASK}_s${s}_single; P=${PORTS[$(((n + OFF) % ${#PORTS[@]}))]}; G=$([ $P -lt 5760 ] && echo 0 || echo 3)
  (for try in 1 2; do
     [ -f out/$tag.json ] && break
     CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G $PY closedloop.py --tag $tag --port $P --out out >> logs/$tag.log 2>&1 \
       && echo "$(date '+%m-%d %H:%M:%S') $tag done" >> run.log || echo "$(date '+%m-%d %H:%M:%S') $tag FAILED try $try" >> run.log
   done) &
  n=$((n + 1))
done
wait
echo "$(date '+%m-%d %H:%M:%S') $TASK all done" >> run.log
