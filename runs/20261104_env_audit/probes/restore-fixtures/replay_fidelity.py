"""Online result vs the two replay protocols that rebuild python state from saved per-step states:
  A (stage_trace.py): set_state + env._check_success() every frame
  B (collect.py replay()): set_state every frame; stage checks only at chunk boundaries (i % 16 == 0)
Reports the first frame where _check_success() holds (A) / the chunk boundary where all stages passed (B), and the task
accumulators at the end. usage: python replay_fidelity.py TASK TAG"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
from atomic_steps import TASKS

ACC = {"WashLettuce": "washed_time", "StirVegetables": "success_time", "ScrubCuttingBoard": "board_contact_timer",
       "RinseSinkBasin": "washed_loc", "GetToastedBread": "toaster_on"}
task, tag = sys.argv[1], sys.argv[2]
row = [r for r in rows() if r["tag"] == tag][0]
env = make_env(task)
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
steps = TASKS[task]
out = dict(task=task, tag=tag, online_success=row["success"], online_steps=row["steps"], online_reached=row["reached"])
# A
load_scene(env, scene_of(row))
fa = None
for f in range(len(states)):
    set_state(env, states[f])
    if env._check_success() and fa is None:
        fa = f
out["A_first_success_frame"] = fa
out["A_acc_end"] = repr(getattr(env, ACC[task]))
# B: exactly collect.py replay(), over the whole episode
load_scene(env, scene_of(row))
S, k, fb = {}, 0, None
for i in range(len(states)):
    set_state(env, states[i])
    if i and i % 16 == 0:
        while k < len(steps) and steps[k]["check"](env, S):
            k += 1
        if k >= len(steps) and fb is None:
            fb = i
out["B_all_stages_at"] = fb
out["B_k_end"] = k
out["B_acc_end"] = repr(getattr(env, ACC[task]))
out["B_success_end"] = bool(env._check_success())
print(json.dumps(out))
