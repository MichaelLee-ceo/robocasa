"""Task-level python state (env attributes, not fixtures) vs restore_node / collect.replay.
usage: python -I test_counters.py TASK TAG T_NODE"""
import sys, os, copy, json
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480", "runs/20261026_imagine", "runs/20261026_critic_data"):
    sys.path.insert(0, f"{R}/{p}")
from analyze_stages import replay_env  # noqa
import atomic_eval  # noqa
from atomic_eval import load_scene, restore_node
from oracle import set_state
import hidden_state
from atomic_steps import TASKS
from paths import local

task, tag, tn = sys.argv[1], sys.argv[2], int(sys.argv[3])
ATTRS = dict(RinseSinkBasin=["washed_loc"], WashLettuce=["washed_time"], StirVegetables=["success_time"],
             ScrubCuttingBoard=["board_contact_timer"], GetToastedBread=["toaster_on"])[task]
rows = [json.loads(l) for p in __import__("glob").glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
row = next(r for r in rows if r["tag"] == tag)
states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
env = replay_env(task)
steps = TASKS[task]


def attrs():
    return {a: copy.deepcopy(getattr(env, a)) for a in ATTRS}


# (a) main-run-faithful replay: update_state (via set_state) + _check_success once per step
load_scene(env, local(row["scene"]))
traj, first_succ = [], None
for i in range(len(states)):
    set_state(env, states[i])
    ok = bool(env._check_success()) if i else False
    if ok and first_succ is None:
        first_succ = i
    traj.append(attrs())
print(f"{tag}: logged success={row['success']} steps={row['steps']} history={row['history']}; faithful replay first success step={first_succ}")
print(f"  faithful attrs at t={tn}: {traj[tn]}   at end: {traj[-1]}")

# (b) collect.py replay() to tn
from collect import replay
load_scene(env, local(row["scene"]))
S0, k_rep = replay(env, states, tn, steps)
print(f"  collect.replay attrs at t={tn}: {attrs()}  (k_replay={k_rep})")

# (c) leak: node at tn (collect.py style: settle + capture), a 'sibling rollout' runs to the end, restore the node
hidden_state.settle(env, states[tn - 1])
H0 = hidden_state.capture(env); ps = hidden_state.py_state(env)
before = attrs(); succ_before = bool(env._check_success())
for i in range(tn + 1, len(states)):  # a sibling rollout that reaches the end of the episode
    set_state(env, states[i]); env._check_success()
hidden_state.set_py_state(env, ps)
restore_node(env, states[tn], H0)
print(f"  node t={tn} [obs=False]: attrs before={before} success_before={succ_before} -> after a sibling + restore_node: "
      f"attrs={attrs()} _check_success()={bool(env._check_success())}")

# (d) same with an obs=True node (imagine / closedloop)
load_scene(env, local(row["scene"]))
for i in range(tn + 1):
    set_state(env, states[i]); env._check_success() if i else None
H1 = hidden_state.capture(env, obs=True); ps1 = hidden_state.py_state(env)
before = attrs()
for i in range(tn + 1, len(states)):
    set_state(env, states[i]); env._check_success()
hidden_state.set_py_state(env, ps1)
restore_node(env, states[tn], H1)
print(f"  node t={tn} [obs=True]: attrs before={before} -> after a sibling + restore_node: attrs={attrs()} "
      f"_check_success()={bool(env._check_success())}")
