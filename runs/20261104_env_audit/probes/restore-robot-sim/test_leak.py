"""Leak of task-level python state between candidates restored from one node (imagine.candidate / closedloop.py /
collect.py pattern: set_py_state(node ps) then rollout -> restore_node(state, hidden)). Candidate 1 = the saved episode's
own continuation (set_state + 2x _check_success per step, as env.step does), candidate 2 = a fresh restore of the node."""
import sys, json, copy
sys.path.insert(0, sys.argv[1])
from harness import *
from atomic_eval import load_scene, restore_node
from oracle import set_state
import hidden_state
from hidden_state import py_state, set_py_state

task, tag, attr, t_node = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
RUN = f"{ROOT}/runs/20261025_seen480/run"
job = next(j for j in json.load(open(f"{RUN}/jobs.json")) if j["tag"] == tag)
env = make_env(task)
load_scene(env, job["scene"])
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
def v(e):
    x = getattr(e, attr)
    return (len(x) if attr == "board_contact_positions" else list(x)) if isinstance(x, list) else x
for i in range(1, t_node + 1):
    set_state(env, states[i]); env._check_success(); env._check_success()
ps = py_state(env); H = hidden_state.capture(env, obs=True)
node_val, node_succ = copy.deepcopy(v(env)), bool(env._check_success())
# candidate 1: the episode's own continuation to its end
for i in range(t_node + 1, len(states)):
    set_state(env, states[i]); env._check_success(); env._check_success()
c1_val, c1_succ = copy.deepcopy(v(env)), bool(env._check_success())
# candidate 2: restore the node exactly as the pipeline does
set_py_state(env, ps); restore_node(env, states[t_node], H)
c2_val, c2_succ = copy.deepcopy(v(env)), bool(env._check_success())
print("RESULT", json.dumps(dict(task=task, tag=tag, attr=attr, t_node=t_node, T=len(states) - 1, node=node_val, node_success=node_succ,
                                cand1_end=c1_val, cand1_success=c1_succ, cand2_start_after_restore=c2_val, cand2_success_at_start=c2_succ,
                                attr_in_py_state=any(attr in d for d in ps.values())), default=str))
