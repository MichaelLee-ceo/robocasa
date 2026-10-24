import sys, os, json, glob, collections
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
from atomic_eval import load_scene
from oracle import set_state, reset_to
from atomic_steps import TASKS, prepare
from demo_utils import load_demo
V2 = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/20261013_scenes_v2"; DEC = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/20261012_atomic_decomp_seen"
def load(d):
    R = {}
    for p in glob.glob(f"{d}/results_w*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r: R[r["tag"]] = r
    return R
RV, RD = load(V2), load(DEC)
NST = dict(KettleBoiling=2, PreSoakPan=3, StackBowlsCabinet=2, SteamInMicrowave=4, PrepareCoffee=2)
pre_fresh, pre_demo = {}, {}
for t, n in NST.items():
    b = gym.make(f"robocasa/{t}", split="target", enable_render=False).unwrapped; e = b.env; steps = TASKS[t]
    for s in range(20):
        f = f"{V2}/{t}/{t}_s{s}_single_states.npz"
        recs = {int(x["k"]): x for x in np.load(f, allow_pickle=True)["recs"]}
        for k in range(1, n):
            if k not in recs: continue
            load_scene(e, f"{os.path.expanduser(os.environ.get('ROBOCASA_ROOT', '~/Desktop/robocasa'))}/runs/scenes_seen5_v2/{t}_s{s}.npz"); set_state(e, recs[k]["state"])
            pre_fresh[(t, s, k)] = bool(steps[k]["check"](e, dict(recs[k]["S"])))
    demos = sorted({r["demo"] for r in RD.values() if r["task"] == t and r["condition"] == "reset_atomic"})
    for dm in demos:
        d = load_demo(t, dm)
        reset_to(e, d["xml"], d["ep_meta"]); S0 = prepare(e, d, set_state); reset_to(e, d["xml"], d["ep_meta"]); e.unset_ep_meta()
        segs = d["segs"]; starts = [segs[st["seg"]]["start"] for st in steps]
        for k in range(1, n):
            set_state(e, d["states"][starts[k]]); pre_demo[(t, dm, k)] = bool(steps[k]["check"](e, dict(S0)))
print("== fresh starts (single hand-off states)")
for t, n in NST.items():
    for k in range(1, n):
        keys = [x for x in pre_fresh if x[0] == t and x[2] == k]; pre = {x[1] for x in keys if pre_fresh[x]}
        line = f"{t} step {k+1} {TASKS[t][k]['skill']}: starts {len(keys)}, already satisfied {len(pre)} {sorted(pre)}"
        for name, pat in (("aseen solo", f"_aseen_solo_k{k}"), ("pre solo", f"_pre_solo_k{k}"), ("home base+arm", f"_aseen_home_k{k}"), ("home arm-only", f"_aseen_armhome_k{k}")):
            rs = [r for tg, r in RV.items() if r["task"] == t and tg.endswith(pat)]
            if not rs: continue
            succ = [r["seed"] for r in rs if r["step_success"]]
            line += f" | {name}: {len(succ)}/{len(rs)}, excl. pre-satisfied {len([x for x in succ if x not in pre])}/{len([r for r in rs if r['seed'] not in pre])}"
        print(line)
print("== demo starts (first frame of the step's segment)")
for t, n in NST.items():
    for k in range(1, n):
        keys = [x for x in pre_demo if x[0] == t and x[2] == k]; pre = {x[1] for x in keys if pre_demo[x]}
        line = f"{t} step {k+1}: demo starts {len(keys)}, already satisfied {len(pre)} {sorted(pre)}"
        for name, R, cond in (("cs atomic", RD, "reset_atomic"), ("cs full", RD, "reset_task"), ("aseen", RV, "reset_atomic_ackpt"), ("pre", RV, "reset_atomic_pckpt")):
            rs = [r for r in R.values() if r["task"] == t and r["condition"] == cond and r.get("step") == k]
            if not rs: continue
            succ = [r["demo"] for r in rs if r["step_success"]]
            line += f" | {name}: {len(succ)}/{len(rs)}, excl. {len([x for x in succ if x not in pre])}/{len([r for r in rs if r['demo'] not in pre])}"
        print(line)
json.dump(dict(fresh={"|".join(map(str, k)): v for k, v in pre_fresh.items()}, demo={"|".join(map(str, k)): v for k, v in pre_demo.items()}),
          open(f"{V2}/presatisfied_starts.json", "w"), indent=1)
