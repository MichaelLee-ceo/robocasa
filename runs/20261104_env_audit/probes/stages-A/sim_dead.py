"""SteamInMicrowave: state of the official conjuncts at the chunk boundaries where the tracker credited stages 1 and 2."""
import sys, json, glob
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/stages-A")
from common import *
from atomic_steps import TASKS
task = "SteamInMicrowave"
rows = {json.loads(l)["seed"]: json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)
        if json.loads(l)["task"] == task}
env = make_env(task)
for s in map(int, sys.argv[1:]):
    load(env, task, s)
    st = np.load(f"{R}/runs/20261025_seen480/run/{task}/{task}_s{s}_single_steps.npz")["states"]
    for t, k in rows[s]["history"] + [[len(st) - 1, "end"]]:
        set_state(env, st[t])
        print(f"s{s} t={t} credited stage {k}: veg_in_bowl={bool(OU.check_obj_in_receptacle(env, 'vegetable', 'bowl'))} "
              f"bowl_in_mw={bool(OU.obj_inside_of(env, 'bowl', env.microwave))} door_closed={bool(env.microwave.is_closed(env=env))} "
              f"veg_z={float(env.sim.data.body_xpos[env.obj_body_id['vegetable']][2]):.3f}", flush=True)
