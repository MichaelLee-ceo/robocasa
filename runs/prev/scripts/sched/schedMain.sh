#!/bin/bash
ROBOCASA_ROOT="${ROBOCASA_ROOT:-$HOME/Desktop/robocasa}"; RC_DIR="${RC_DIR:-/tmp/claude-1001/rc}"; export ROBOCASA_ROOT RC_DIR
mkdir -p "$RC_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
R=$ROBOCASA_ROOT/runs; OUT=$R/20261013_scenes_main; S=$R/scripts/atomic_eval.py; CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning
nrun() { ps -eo args | grep -c "^python -I .*atomic_eval.py --jobs $1"; }
server() { # $1 model path, $2 task_set; only one policy server at a time
  for p in $(ps -eo pid,args | awk '$2 ~ /python$/ && /run_eval.py --server/{print $1}'); do kill $p; done
  while ss -ltn | grep -q ":5555 "; do sleep 3; done
  (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && nohup setsid python scripts/run_eval.py --server --model_path $1 --task_set $2 --split target --port 5555 > $RC_DIR/server_$2.log 2>&1 < /dev/null &)
  until ss -ltn | grep -q ":5555 "; do sleep 5; done; }
runjobs() { cd $OUT; for w in $(seq 0 5); do python -I $S --jobs $1 --worker $w --n_workers 6 --out $OUT --port 5555 > worker_${1%.json}_$w.log 2>&1 & done; wait
  # one retry pass for errored jobs (e.g. render OOM)
  for w in $(seq 0 5); do python -I $S --jobs $1 --worker $w --n_workers 6 --out $OUT --port 5555 >> worker_${1%.json}_$w.log 2>&1 & done; wait; }
while [ $(nrun jobs_single.json) -gt 0 ]; do sleep 20; done
echo "$(date +%H:%M) single pass1 done"; runjobs jobs_single.json; echo "$(date +%H:%M) single done"
python $R/scripts/gen_jobs_main.py _ackpt aseen 1
server $CK/target_posttraining/atomic_seen/checkpoint-60000 atomic_seen; echo "$(date +%H:%M) aseen server up"
runjobs jobs_aseen.json; echo "$(date +%H:%M) aseen done"
python $R/scripts/gen_jobs_main.py _pckpt pre 0
server $CK/pretraining/checkpoint-80000 atomic_seen; echo "$(date +%H:%M) pre server up"
mkdir -p $R/20261013_pretrain_sanity; cd $R/20261013_pretrain_sanity
sed "s#20261012_atomic_seen_sanity#20261013_pretrain_sanity#; s#5556#5555#" $R/scripts/atomic_seen_sanity.py > $R/scripts/pretrain_sanity.py
python $R/scripts/pretrain_sanity.py PickPlaceCounterToStove,PickPlaceCounterToCabinet,TurnOnMicrowave 10 > sanity.log 2>&1
echo "$(date +%H:%M) pre sanity done: $(cat $R/20261013_pretrain_sanity/evals/target/*/stats.json | grep -o 'success_rate": [0-9.]*' | tr '\n' ' ')"
echo "WAITING_FOR_GO_PRE"
until [ -f $RC_DIR/go_pre ]; do sleep 20; done
runjobs jobs_pre.json; echo "$(date +%H:%M) pre done"
