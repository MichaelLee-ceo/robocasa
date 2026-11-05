"""collect.py restore path at a settled node: ps from replay to t, H0 = capture(kenv) (no obs); before every rollout
set_py_state(ps) then restore_node(state, H0). Does the node's python state survive, and is it stable across rollouts?"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
task, tag, t = sys.argv[1], sys.argv[2], int(sys.argv[3])
row = [r for r in rows() if r["tag"] == tag][0]
env = make_env(task); load_scene(env, scene_of(row))
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
for i in range(t + 1):  # collect.py replay(): set_state only
    set_state(env, states[i])
def look(e):
    d = {}
    if task == "GetToastedBread":
        d = dict(num_steps_on=dict(e.toaster._num_steps_on), turned_on=dict(e.toaster._turned_on))
    if task == "WashLettuce":
        d = dict(washed_time=e.washed_time)
    return d
ps, H0 = py_state(env), hidden_state.capture(env)
out = dict(task=task, t=t, after_replay=look(env), rollouts=[])
for m in range(3):
    set_py_state(env, ps); restore_node(env, states[t], H0)
    out["rollouts"].append(look(env))
print(json.dumps(out))
