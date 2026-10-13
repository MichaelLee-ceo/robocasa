#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ~/Desktop/robocasa/envs/groot
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$HOME/Desktop/robocasa/.hf_cache/hub
R=~/Desktop/robocasa/runs; OUT=$R/20261013_scenes_v2; S=$R/scripts/atomic_eval.py; CK=~/Desktop/robocasa/checkpoints/gr00t_n1-5/foundation_model_learning; P=5556
mkdir -p $OUT
server() { # only one policy server at a time
  for p in $(ps -eo pid,args | awk '$2 ~ /python$/ && /run_eval.py --server/{print $1}'); do kill $p; done
  while ss -ltn | grep -q ":$P "; do sleep 3; done
  (cd ~/Desktop/robocasa/third_party/Isaac-GR00T && nohup setsid python scripts/run_eval.py --server --model_path $1 --task_set $2 --split target --port $P > /tmp/claude-1001/rc/server_v2_$2.log 2>&1 < /dev/null &)
  until ss -ltn | grep -q ":$P "; do sleep 5; done; }
runjobs() { cd $OUT; for pass in 1 2; do for w in $(seq 0 5); do python -I $S --jobs $1 --worker $w --n_workers 6 --out $OUT --port $P >> worker_${1%.json}_$w.log 2>&1 & done; wait; done; }
python3 - <<'PY'
import json
H=json.load(open('/home/michaellee/Desktop/robocasa/runs/20261012_groot_repro_seen_h1x/horizons_original.json'))
SC='/home/michaellee/Desktop/robocasa/runs/scenes_seen5_v2'
T=["KettleBoiling","PreSoakPan","StackBowlsCabinet","SteamInMicrowave","PrepareCoffee"]
json.dump([dict(task=t,seed=s,scene=f"{SC}/{t}_s{s}.npz",condition="baseline_seed_save",approach="single",horizon=H[t],tag=f"{t}_s{s}_single") for s in range(20) for t in T],
          open('/home/michaellee/Desktop/robocasa/runs/20261013_scenes_v2/jobs_single.json','w'))
PY
ss -ltn | grep -q ":$P " || server $CK/target_posttraining/composite_seen/checkpoint-60000 composite_seen
echo "$(date +%H:%M) single start"; runjobs jobs_single.json; echo "$(date +%H:%M) single done"
echo "WAITING_FOR_GO_ASEEN"; until [ -f /tmp/claude-1001/rc/go_aseen ]; do sleep 20; done
python $R/scripts/gen_jobs_main.py _ackpt aseen 1
server $CK/target_posttraining/atomic_seen/checkpoint-60000 atomic_seen; echo "$(date +%H:%M) aseen server up"
runjobs jobs_aseen.json; echo "$(date +%H:%M) aseen done"
python $R/scripts/gen_jobs_main.py _pckpt pre 0
server $CK/pretraining/checkpoint-80000 atomic_seen; echo "$(date +%H:%M) pre server up"
mkdir -p $R/20261013_pretrain_sanity; cd $R/20261013_pretrain_sanity
sed "s#20261012_atomic_seen_sanity#20261013_pretrain_sanity#" $R/scripts/atomic_seen_sanity.py > $R/scripts/pretrain_sanity.py
python $R/scripts/pretrain_sanity.py PickPlaceCounterToStove,PickPlaceCounterToCabinet,TurnOnMicrowave 10 > sanity.log 2>&1
echo "$(date +%H:%M) pre sanity done: $(cat $R/20261013_pretrain_sanity/evals/target/*/stats.json | grep -o 'success_rate": [0-9.]*' | tr '\n' ' ')"
echo "WAITING_FOR_GO_PRE"; until [ -f /tmp/claude-1001/rc/go_pre ]; do sleep 20; done
runjobs jobs_pre.json; echo "$(date +%H:%M) pre done"
