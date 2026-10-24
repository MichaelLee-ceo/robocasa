"""Job lists on the frozen scenes for one checkpoint suffix (_ackpt Atomic-Seen, _pckpt pretrain-only)."""
import json, sys, os
import numpy as np
sfx, name, with_full = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
OUT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/20261013_scenes_v2")
H = json.load(open(os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/20261012_groot_repro_seen_h1x/horizons_original.json")))
SC = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/scenes_seen5_v2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NST = dict(KettleBoiling=2, PreSoakPan=3, StackBowlsCabinet=2, SteamInMicrowave=4, PrepareCoffee=2)
T = list(NST)
jobs = [dict(task=t, seed=s, scene=f"{SC}/{t}_s{s}.npz", condition="chain_seed" + sfx, approach=name, horizon=H[t],
             tag=f"{t}_s{s}_{name}_chain") for s in range(20) for t in T]
for s in range(20):
    for t in T:
        src = f"{t}_s{s}_single"
        have = {0}
        try:
            have |= {int(x["k"]) for x in np.load(f"{OUT}/{t}/{src}_states.npz", allow_pickle=True)["recs"]}
        except Exception:
            pass
        for k in range(NST[t]):
            if k in have:
                jobs.append(dict(task=t, seed=s, step=k, scene=f"{SC}/{t}_s{s}.npz", condition="reset_seed_atomic" + sfx, approach=name,
                                 horizon=H[t], src_tag=src, tag=f"{t}_s{s}_{name}_solo_k{k}"))
for j in json.load(open(os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/20261012_atomic_decomp_seen/jobs.json"))):
    if j["condition"] == "reset_atomic":
        jobs.append(dict(j, condition="reset_atomic" + sfx, approach=name, tag=j["tag"].replace("reset_atomic", f"{name}_solodemo")))
if with_full:
    jobs += [dict(task=t, seed=s, scene=f"{SC}/{t}_s{s}.npz", condition="baseline_seed" + sfx, approach=name, horizon=H[t],
                  tag=f"{t}_s{s}_{name}_full") for s in range(20) for t in T]
json.dump(jobs, open(f"{OUT}/jobs_{name}.json", "w")); print(name, len(jobs))
