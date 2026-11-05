"""Skeptic test of 'task-level python success state leaks between branches' with real dynamics, no manual setattr.
StirVegetables: replay a seen480 episode (set_state per step = update_state per step, as collect.py replay()) to a
node at a chunk boundary mid-stir, capture as collect.py / imagine.py do, branch A = continue the replay to the end,
then restore the node through the real atomic_eval.restore_node and take ONE env.step with a zero action: is the
official per-step reward (what rollout() calls success) 1 although the node's own counter is < 5?
usage: python -I leak.py stir <seed> | toast <seed>"""
import json, sys, glob
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts"); sys.path.insert(0, f"{R}/third_party/Isaac-GR00T")
import robosuite  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
import hidden_state  # noqa
from hidden_state import py_state, set_py_state  # noqa
from atomic_eval import restore_node  # noqa  (the real one)
from atomic_steps import TASKS  # noqa
from paths import local  # noqa

which, seed = sys.argv[1], int(sys.argv[2])
task = {"stir": "StirVegetables", "toast": "GetToastedBread"}[which]
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
r = [x for x in rows if x["task"] == task and x["seed"] == seed and "error" not in x][0]
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
z = np.load(local(r["scene"]), allow_pickle=True)
states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
zero = np.zeros(env.action_spec[0].shape)


def fresh():
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    for v in ("veg1", "veg2"):
        if hasattr(env, f"prev_obj_pos_{v}"):
            delattr(env, f"prev_obj_pos_{v}")


def counter():
    return env.success_time if which == "stir" else env.toaster_on


def replay_to(t, per_step_check):
    trace = []
    for i in range(t + 1):
        set_state(env, states[i])
        if per_step_check:
            env._check_success()
        trace.append(counter())
    return trace


out = dict(tag=r["tag"], live_success=r["success"], live_steps=r["steps"])
fresh()
trace = replay_to(len(states) - 1, per_step_check=(which == "toast"))
if which == "stir":
    incs = [i for i in range(1, len(trace)) if trace[i] != trace[i - 1]]
    out["replay_increments_at"] = incs; out["replay_end"] = trace[-1]
    # node: last chunk boundary with 1 <= counter <= 3 (mid-stir), else the last boundary before the first increment
    cands = [t for t in range(16, len(states), 16) if 1 <= trace[t] <= 3]
    t_node = cands[-1] if cands else (max(16, (incs[0] // 16) * 16) if incs else 16 * ((len(states) - 1) // 32))
else:
    on = [i for i, v in enumerate(trace) if v]
    out["replay_latch_at"] = on[0] if on else None
    t_node = 0
out["t_node"] = t_node

# continuous reference: fresh env, replay to node, one zero-action step
fresh(); replay_to(t_node, which == "toast")
if t_node == 0:
    pass
h = hidden_state.capture(env, obs=True); ps = py_state(env)
out["node_value"] = counter() if not isinstance(counter(), np.ndarray) else None
node_val = counter()
_, rew_cont, _, _ = env.step(zero)
out["continuous_first_step"] = dict(value=counter(), reward=float(rew_cont))

# branch A: from the same fresh replay-to-node, continue the replay to the end of the episode
fresh(); replay_to(t_node, which == "toast")
h = hidden_state.capture(env, obs=True); ps = py_state(env)
for i in range(t_node + 1, len(states)):
    set_state(env, states[i])
    if which == "toast":
        env._check_success()
out["after_branch_A"] = counter()
val_A = counter()
# branch B: restore the node exactly as imagine.candidate / closedloop / collect do
set_py_state(env, ps)
restore_node(env, states[t_node], h)
out["after_restore_node"] = dict(value=counter(), check_success=bool(env._check_success()),
                                 delta_vs_A_end=(int(counter()) - int(val_A)) if which == "stir" else None)
_, rew_B, _, _ = env.step(zero)
out["branch_B_first_step"] = dict(value=counter(), reward=float(rew_B))
print(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), flush=True)
