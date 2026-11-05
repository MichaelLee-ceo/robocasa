"""RinseSinkBasin: washed_loc rebuilt the way collect.py replay() does (set_state per frame, stage checks only at chunk
boundaries) vs the live rule (_check_success every control step), at every chunk boundary of seen480 episodes."""
import sys
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/stages-A")
from common import *
from atomic_steps import TASKS
task = "RinseSinkBasin"
env = make_env(task)
steps = TASKS[task]
for s in map(int, sys.argv[1:]):
    load(env, task, s)
    st = np.load(f"{R}/runs/20261025_seen480/run/{task}/{task}_s{s}_single_steps.npz")["states"]
    # live rule: washed_loc latched at every frame where _check_success runs
    live = [False] * 3; S, k = {}, 0; first_diff = None; live_succ = None; rep_succ = None
    env.washed_loc = [False] * 3
    rep_loc = None
    seq = []
    for i in range(len(st)):
        set_state(env, st[i])
        hs = env.sink.get_handle_state(env=env)
        if hs["water_on"]:
            live[{"left": 0, "center": 1, "right": 2}[hs["spout_ori"]]] = True
        seq.append((bool(hs["water_on"]), hs["spout_ori"]))
        if i and i % 16 == 0:   # collect.replay(): stage checks only here
            while k < len(steps) and steps[k]["check"](env, S):
                k += 1
            if live_succ is None and all(live):
                live_succ = i
            if rep_succ is None and all(env.washed_loc):
                rep_succ = i
            if first_diff is None and list(env.washed_loc) != live:
                first_diff = (i, list(env.washed_loc), list(live))
    print(f"s{s}: live all-washed at boundary {live_succ}, replay() all-washed at {rep_succ}, first boundary where replay != live: {first_diff}", flush=True)
