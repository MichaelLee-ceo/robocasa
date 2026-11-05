import sys, copy, json, glob
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480", "runs/20261026_imagine", "runs/20261026_critic_data"):
    sys.path.insert(0, f"{R}/{p}")
from analyze_stages import replay_env
import atomic_eval
from atomic_eval import load_scene
from oracle import set_state
from atomic_steps import TASKS
from paths import local
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
task = "RinseSinkBasin"; steps = TASKS[task]
env = replay_env(task)
n_diff = 0; n_tot = 0
for row in [r for r in rows if r["task"] == task]:
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{row['tag']}_steps.npz")["states"]
    load_scene(env, local(row["scene"]))
    faith, k, S, coll = [], 0, {}, []
    # faithful: per step _check_success; collect-style: _check_success only via the stage checks at chunk boundaries
    for i in range(len(states)):
        set_state(env, states[i]); env._check_success() if i else None
        faith.append(list(env.washed_loc))
    load_scene(env, local(row["scene"]))
    for i in range(len(states)):
        set_state(env, states[i])
        if i and i % 16 == 0:
            while k < len(steps) and steps[k]["check"](env, S):
                k += 1
        coll.append(list(env.washed_loc))
    d = [i for i in range(0, len(states), 16) if faith[i] != coll[i]]
    n_tot += 1; n_diff += bool(d)
    if d:
        i = d[0]
        print(row["tag"], "success", row["success"], "first boundary with a different washed_loc", i, "faithful", faith[i], "collect.replay", coll[i], "n boundaries differing", len(d))
print(f"{n_diff}/{n_tot} episodes: collect.replay's washed_loc differs from the main run's at some chunk boundary")
