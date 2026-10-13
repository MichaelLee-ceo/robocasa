#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ~/Desktop/robocasa/envs/groot
export NO_ALBUMENTATIONS_UPDATE=1
SD=~/Desktop/robocasa/runs/20261012_atomic_seeded; S=~/Desktop/robocasa/runs/scripts/atomic_eval.py
while [ $(ps -eo args | grep -c "^python -I .*jobs_AB.json") -gt 0 ]; do sleep 20; done
echo "$(date +%H:%M) A/B done, starting D1"
cd $SD; for w in $(seq 0 5); do python -I $S --jobs jobs_D1.json --worker $w --n_workers 6 --out $SD --port 5555 > worker_D1_$w.log 2>&1 & done; wait
echo "$(date +%H:%M) D1 done"
python3 - <<'PY'
import json, glob, numpy as np
SD='/home/michaellee/Desktop/robocasa/runs/20261012_atomic_seeded'
H=json.load(open('/home/michaellee/Desktop/robocasa/runs/20261012_groot_repro_seen_h1x/horizons_original.json'))
want={"StackBowlsCabinet":[1],"SteamInMicrowave":[1,2,3]}
jobs=[]
for t,ks in want.items():
    for s in range(20):
        src=f"{t}_s{s}_baseline_seed_save"; f=f"{SD}/{t}/{src}_states.npz"
        try: recs=np.load(f,allow_pickle=True)["recs"]
        except Exception: continue
        have={int(x["k"]) for x in recs}
        for k in ks:
            if k in have:
                for c in ("reset_seed_atomic","reset_seed_task"):
                    jobs.append(dict(task=t,seed=s,step=k,condition=c,horizon=H[t],src_tag=src,tag=f"{t}_s{s}_{c}_k{k}"))
json.dump(jobs,open(f"{SD}/jobs_D2.json","w")); print("D2 jobs",len(jobs))
PY
cd $SD; for w in $(seq 0 5); do python -I $S --jobs jobs_D2.json --worker $w --n_workers 6 --out $SD --port 5555 > worker_D2_$w.log 2>&1 & done; wait
echo "$(date +%H:%M) D2 done"
