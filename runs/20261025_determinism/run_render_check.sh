#!/bin/bash
# render_check.py on GPU0: two solo runs one after the other, then two runs at the same time (same seeded server).
set -u
export ROBOCASA_ROOT=/home/michaellee/mclee/robocasa
source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PIP_CONFIG_FILE=/dev/null
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
D=$ROBOCASA_ROOT/runs/20261025_determinism/render_check; P=5585; N=${1:-40}
mkdir -p $D
if ss -ltn | grep -q ":$P "; then echo "port $P is busy"; exit 1; fi
(cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && exec nice -n 10 python $ROBOCASA_ROOT/runs/scripts/serve_seeded.py \
   --model_path $CK --port $P) > $D/server.log 2>&1 < /dev/null &
SRV=$!
until ss -ltn | grep -q ":$P "; do kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
RC="nice -n 10 python -I $ROBOCASA_ROOT/runs/20261025_determinism/render_check.py $P $N"
$RC $D/solo_a > $D/solo_a.log 2>&1; tail -1 $D/solo_a.log
$RC $D/solo_b > $D/solo_b.log 2>&1; tail -1 $D/solo_b.log
$RC $D/par_a > $D/par_a.log 2>&1 & A=$!
$RC $D/par_b > $D/par_b.log 2>&1 & B=$!
wait $A $B; tail -1 $D/par_a.log; tail -1 $D/par_b.log
kill $SRV
echo done
