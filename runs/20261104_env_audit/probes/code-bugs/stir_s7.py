import sys, json, glob
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480"):
    sys.path.insert(0, f"{R}/{p}")
from analyze_stages import replay_env
import atomic_eval  # noqa
from atomic_eval import load_scene
from oracle import set_state
from paths import local
tag = "StirVegetables_s7_single"
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
row = next(r for r in rows if r["tag"] == tag)
states = np.load(f"{R}/runs/20261025_seen480/run/StirVegetables/{tag}_steps.npz")["states"]
env = replay_env("StirVegetables")
load_scene(env, local(row["scene"]))
inc = []
for i in range(len(states)):
    a = env.success_time
    set_state(env, states[i])
    if env.success_time > a:
        inc.append(i)
print(tag, "live: success at step <=", row["steps"], "| replay success_time at end:", env.success_time, "increments at steps", inc)
