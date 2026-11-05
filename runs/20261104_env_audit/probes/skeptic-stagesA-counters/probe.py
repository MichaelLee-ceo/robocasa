"""Skeptic probe: are task-level counters (env attributes) part of a restored node?
Branch = replay of a saved, successful plain seen480 episode (set_state per step runs update_state, as env.step's
_post_action does; _check_success per step, as the gym wrapper's step does). Then restore an earlier node exactly as
closedloop.py does (set_py_state + restore_node) and check the counters and the official check."""
import os, sys, json, copy
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts"); sys.path.insert(0, f"{R}/third_party/Isaac-GR00T")
import robosuite, robocasa  # noqa
from robocasa.utils.env_utils import load_composite_controller_config
from oracle import reset_to, set_state
from paths import local
import hidden_state
from hidden_state import py_state, set_py_state
from atomic_eval import restore_node

task, s = sys.argv[1], int(sys.argv[2])
cc = load_composite_controller_config(controller=None, robot="PandaOmron")
env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cc, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, use_object_obs=True,
                     translucent_robot=False, obj_instance_split="target",
                     layout_and_style_ids=list(zip(range(1, 11), range(1, 11))))
z = np.load(f"{R}/runs/scenes_seen11_v2/{task}_s{s}.npz", allow_pickle=True)
reset_to(env, local(str(z["xml"])), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
s0 = z["state"].copy()
ATTR = {"WashLettuce": ["washed_time"], "RinseSinkBasin": ["washed_loc"],
        "ScrubCuttingBoard": ["board_contact_timer", "board_contact_positions"]}[task]
def C():
    out = {}
    for a in ATTR:
        v = getattr(env, a)
        out[a] = len(v) if a == "board_contact_positions" else copy.deepcopy(v)
    return out
print("load_scene:", C(), "check", env._check_success())
node0 = dict(state=s0, ps=py_state(env), hidden=hidden_state.capture(env))  # closedloop.py:57 / imagine.py:105
states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{task}_s{s}_single_steps.npz")["states"]
first = None; first_inc = None; prev = C()
for i in range(1, len(states)):
    set_state(env, states[i])
    if env._check_success() and first is None:
        first = i
    if first_inc is None and C() != prev:
        first_inc = i
print(f"branch replayed ({len(states)-1} steps): counters {C()}, official success first at step {first}, counter first changed at {first_inc}")
# restore node0 as closedloop.py:66-67 does
set_py_state(env, node0["ps"]); restore_node(env, node0["state"], node0["hidden"])
print("after restore of node0 (t=0):", C(), "| sim==node:", bool(np.array_equal(env.sim.get_state().flatten(), s0)),
      "| official check:", env._check_success())
a = np.zeros(env.action_spec[0].shape)
env.step(a)
print("one env.step from t=0 -> gym-wrapper reward (_check_success after step):", env._check_success())

if task == "WashLettuce" and first_inc is not None:
    # node reached at the first under-water step: capture(obs=True) as closedloop.py:88 does, then restore it repeatedly
    reset_to(env, local(str(z["xml"])), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    for i in range(1, first_inc + 1):
        set_state(env, states[i])
    t = first_inc
    nd = dict(state=states[t].copy(), ps=py_state(env), hidden=hidden_state.capture(env, obs=True))
    w0 = env.washed_time
    seq = []
    for r in range(5):
        set_py_state(env, nd["ps"]); restore_node(env, nd["state"], nd["hidden"]); seq.append(env.washed_time)
    print(f"node at t={t} (lettuce under water), washed_time at capture {w0}; after each of 5 restores: {seq}")
    # a 16-step candidate from that node, then restore: what the next candidate starts with
    for i in range(t + 1, t + 17):
        set_state(env, states[i])
    after_cand = env.washed_time
    set_py_state(env, nd["ps"]); restore_node(env, nd["state"], nd["hidden"])
    print(f"after a 16-step candidate: {after_cand}; the next candidate starts with washed_time {env.washed_time} (node had {w0})")
