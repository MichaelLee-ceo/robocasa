"""RinseSinkBasin / GetToastedBread: the latch (washed_loc / toaster_on) at every chunk boundary under protocol A
(set_state + _check_success every frame = what the live env does) vs protocol B (collect.py replay(): set_state every
frame, stage checks at boundaries). Counts the decision points where B's python state differs from A's.
usage: python rinse_boundaries.py TASK TAG"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
from atomic_steps import TASKS

task, tag = sys.argv[1], sys.argv[2]
attr = {"RinseSinkBasin": "washed_loc", "GetToastedBread": "toaster_on"}[task]
row = [r for r in rows() if r["tag"] == tag][0]
env = make_env(task)
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
steps = TASKS[task]
load_scene(env, scene_of(row))
A = {}
for f in range(len(states)):
    set_state(env, states[f]); env._check_success()
    if f % 16 == 0:
        A[f] = repr(getattr(env, attr))
load_scene(env, scene_of(row))
B, S, k = {}, {}, 0
for i in range(len(states)):
    set_state(env, states[i])
    if i and i % 16 == 0:
        while k < len(steps) and steps[k]["check"](env, S):
            k += 1
    if i % 16 == 0:
        B[i] = repr(getattr(env, attr))
diff = [f for f in A if A[f] != B[f]]
print(json.dumps(dict(task=task, tag=tag, boundaries=len(A), differ=len(diff),
                      first_differ=diff[:1], example=(A[diff[0]], B[diff[0]]) if diff else None)))
