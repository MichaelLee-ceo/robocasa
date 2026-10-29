#!/bin/bash
# Closed-loop oracle steering, KettleBoiling pilot (2026-10-29): 10 scenes where the plain policy often fails and imagine
# succeeds. One process per episode on the GPU0 / GPU3 policy servers (servers.sh), detached; a failed episode is retried
# once. Results: out/<tag>.json; logs: logs/<tag>.log; run.log.
cd "$(dirname "$0")"
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_CONFIG_FILE=/dev/null NO_ALBUMENTATIONS_UPDATE=1
PY="/home/michaellee/mclee/robocasa/envs/groot/bin/python -I"
mkdir -p out logs
SCN=(2 4 5 8 9 10 12 17 19 24); PORTS=(5751 5761 5752 5762 5753 5763 5754 5764 5751 5761)
for n in "${!SCN[@]}"; do
  tag=KettleBoiling_s${SCN[$n]}_single; P=${PORTS[$n]}; G=$([ $P -lt 5760 ] && echo 0 || echo 3)
  (for try in 1 2; do
     [ -f out/$tag.json ] && break
     CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G $PY closedloop.py --tag $tag --port $P --out out >> logs/$tag.log 2>&1 \
       && echo "$(date '+%H:%M:%S') $tag done" >> run.log || echo "$(date '+%H:%M:%S') $tag FAILED try $try" >> run.log
   done) &
done
wait
echo "$(date '+%H:%M:%S') all done" >> run.log
