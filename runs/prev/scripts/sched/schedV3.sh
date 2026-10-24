#!/bin/bash
ROBOCASA_ROOT="${ROBOCASA_ROOT:-$HOME/Desktop/robocasa}"; RC_DIR="${RC_DIR:-/tmp/claude-1001/rc}"; export ROBOCASA_ROOT RC_DIR
mkdir -p "$RC_DIR"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate $ROBOCASA_ROOT/envs/groot
export NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_CACHE=$ROBOCASA_ROOT/.hf_cache/hub
R=$ROBOCASA_ROOT/runs; OUT=$R/20261013_pretrain3; S=$R/scripts/atomic_eval.py; CK=$ROBOCASA_ROOT/checkpoints/gr00t_n1-5/foundation_model_learning; P=5556
mkdir -p $OUT
server() {
  for p in $(ps -eo pid,args | awk '$2 ~ /python$/ && /run_eval.py --server/{print $1}'); do kill $p; done
  while ss -ltn | grep -q ":$P "; do sleep 3; done
  (cd $ROBOCASA_ROOT/third_party/Isaac-GR00T && nohup setsid python scripts/run_eval.py --server --model_path $1 --task_set $2 --split target --port $P > $RC_DIR/server_v3_$2.log 2>&1 < /dev/null &)
  until ss -ltn | grep -q ":$P "; do sleep 5; done; }
runjobs() { cd $OUT; for pass in 1 2; do for w in $(seq 0 5); do python -I $S --jobs $1 --worker $w --n_workers 6 --out $OUT --port $P >> worker_${1%.json}_$w.log 2>&1 & done; wait; done; }
python3 - <<'PY'
import json, os
H=json.load(open(os.environ['ROBOCASA_ROOT']+'/runs/horizons_original_plus.json'))
SC=os.environ['ROBOCASA_ROOT']+'/runs/scenes_seen5_v2'; OUT=os.environ['ROBOCASA_ROOT']+'/runs/20261013_pretrain3'
T=["ArrangeVegetables","GatherVegetables","CondimentCollection"]
json.dump([dict(task=t,seed=s,scene=f"{SC}/{t}_s{s}.npz",condition="baseline_seed_save",approach="single",horizon=H[t],tag=f"{t}_s{s}_single") for s in range(20) for t in T], open(f"{OUT}/jobs_single.json","w"))
PY
server $CK/target_posttraining/composite_seen/checkpoint-60000 composite_seen; echo "$(date +%H:%M) cs server up"
runjobs jobs_single.json; echo "$(date +%H:%M) single done"
python3 - <<'PY'
import json, numpy as np, os
H=json.load(open(os.environ['ROBOCASA_ROOT']+'/runs/horizons_original_plus.json'))
SC=os.environ['ROBOCASA_ROOT']+'/runs/scenes_seen5_v2'; OUT=os.environ['ROBOCASA_ROOT']+'/runs/20261013_pretrain3'
T=["ArrangeVegetables","GatherVegetables","CondimentCollection"]
jobs=[dict(task=t,seed=s,scene=f"{SC}/{t}_s{s}.npz",condition="chain_seed_ackpt",approach="aseen",horizon=H[t],tag=f"{t}_s{s}_aseen_chain") for s in range(20) for t in T]
for s in range(20):
    for t in T:
        src=f"{t}_s{s}_single"; f=f"{OUT}/{t}/{src}_states.npz"
        if os.path.exists(f) and 1 in {int(x["k"]) for x in np.load(f,allow_pickle=True)["recs"]}:
            jobs.append(dict(task=t,seed=s,step=1,scene=f"{SC}/{t}_s{s}.npz",condition="reset_seed_atomic_ackpt",approach="aseen",horizon=H[t],src_tag=src,tag=f"{t}_s{s}_aseen_solo_k1"))
jobs+=[dict(task=t,seed=s,scene=f"{SC}/{t}_s{s}.npz",condition="baseline_seed_ackpt",approach="aseen",horizon=H[t],tag=f"{t}_s{s}_aseen_full") for s in range(20) for t in T]
json.dump(jobs,open(f"{OUT}/jobs_aseen.json","w")); print("aseen jobs",len(jobs))
PY
server $CK/target_posttraining/atomic_seen/checkpoint-60000 atomic_seen; echo "$(date +%H:%M) aseen server up"
runjobs jobs_aseen.json; echo "$(date +%H:%M) aseen done"
for p in $(ps -eo pid,args | awk '$2 ~ /python$/ && /run_eval.py --server/{print $1}'); do kill $p; done; echo "$(date +%H:%M) server stopped"
