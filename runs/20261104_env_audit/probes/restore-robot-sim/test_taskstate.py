"""Task-level python state (env attributes, not fixtures) vs the restore path used by imagine.py / closedloop.py /
collect.py. Uses a successful seen480 episode's saved per-step states."""
import sys, json, copy
sys.path.insert(0, sys.argv[1])
from harness import *
from atomic_eval import load_scene, restore_node
from oracle import set_state
import hidden_state
from hidden_state import py_state, set_py_state

task, tag, attr, target = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
RUN = f"{ROOT}/runs/20261025_seen480/run"
job = next(j for j in json.load(open(f"{RUN}/jobs.json")) if j["tag"] == tag)
env = make_env(task)
load_scene(env, job["scene"])
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
def val(e):
    v = getattr(e, attr)
    return len(v) if isinstance(v, list) and attr == "board_contact_positions" else (list(v) if isinstance(v, list) else v)
def num(e):
    v = getattr(e, attr)
    return len(v) if isinstance(v, list) and attr == "board_contact_positions" else (sum(v) if isinstance(v, list) else float(v))
t_hit = None
for i in range(1, len(states)):  # states[0] = start (load_scene already ran update_state on it)
    set_state(env, states[i]); env._check_success(); env._check_success()  # reward() + gym wrapper, as env.step does
    if num(env) >= target:
        t_hit = i; break
print("hit", attr, val(env), "at step", t_hit, "of", len(states) - 1)
ps = py_state(env); H = hidden_state.capture(env, obs=True)
print("attr in py_state:", any(attr in d for d in ps.values()), " in capture:", attr in H or any(attr in str(k) for k in H))
v_node = copy.deepcopy(getattr(env, attr))
res = dict(task=task, attr=attr, node_value=val(env))
seq = []
for rep in range(3):  # three candidates restored from the same node (imagine.candidate / closedloop.py pattern)
    set_py_state(env, ps)
    restore_node(env, states[t_hit], H)
    seq.append(val(env))
res["after_each_restore"] = seq
# a previous candidate left the counter further along: does the restore put it back?
setattr(env, attr, copy.deepcopy(v_node))
cs_before = bool(env._check_success())
res["success_at_node_with_true_value"] = cs_before
print("RESULT", json.dumps(res, default=str))
