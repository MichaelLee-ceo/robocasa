"""Skeptic re-check: env-level task attributes across restore_node (obs=True node as imagine/closedloop, obs=False settled
node as collect). usage: check.py TASK TAG T_NODE ATTR"""
import sys, json, glob, copy
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480"):
    sys.path.insert(0, f"{R}/{p}")
from analyze_stages import replay_env
import atomic_eval
from atomic_eval import load_scene, restore_node
from oracle import set_state
import hidden_state
from paths import local

task, tag, tn, attr = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
row = next(r for r in rows if r["tag"] == tag)
states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
env = replay_env(task)
g = lambda: copy.deepcopy(getattr(env, attr))
def step_to(i):  # a live step: update_state (via set_state) and reward() -> _check_success()
    set_state(env, states[i]); return bool(env._check_success())

load_scene(env, local(row["scene"]))
for i in range(1, tn + 1):
    step_to(i)
a_node, s_node = g(), bool(env._check_success())
H = hidden_state.capture(env, obs=True); ps = hidden_state.py_state(env)
out = dict(tag=tag, logged_success=row["success"], steps=row["steps"], t=tn, node=str(a_node), node_success=s_node)
# 1. restore with no sibling in between
hidden_state.set_py_state(env, ps); restore_node(env, states[tn], H)
out["restore_only"] = str(g())
setattr(env, attr, copy.deepcopy(a_node))
# 2. a sibling continues to the end of the episode, then the node is restored as imagine/closedloop do
for i in range(tn + 1, len(states)):
    step_to(i)
out["sibling_end"] = str(g())
hidden_state.set_py_state(env, ps); restore_node(env, states[tn], H)
out["after_restore"] = str(g()); out["after_restore_success"] = bool(env._check_success())
# control: put the attribute back to the node's value
setattr(env, attr, copy.deepcopy(a_node)); out["control_success"] = bool(env._check_success())
print("RESULT", json.dumps(out), flush=True)
