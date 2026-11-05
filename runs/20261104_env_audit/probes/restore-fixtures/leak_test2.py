"""Does the node protocol of closedloop.py / imagine.py (node = sim state + py_state + capture(obs=True); restore =
set_py_state + restore_node) put back the task-level state that _check_success depends on?
Replays a saved baseline episode (set_state + _check_success per frame, like stage_trace.py) to a chunk boundary inside
the accumulation window, captures the node, plays the rest of the episode (stands in for a candidate rollout), restores
the node and compares. usage: python leak_test.py TASK TAG"""
import sys, json, copy
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa

TA = {"WashLettuce": ["washed_time"], "StirVegetables": ["success_time", "prev_obj_pos_veg1", "prev_obj_pos_veg2"],
      "ScrubCuttingBoard": ["board_contact_timer", "board_contact_positions"], "RinseSinkBasin": ["washed_loc"],
      "GetToastedBread": ["toaster_on"], "SteamInMicrowave": []}
FX = {"GetToastedBread": "toaster", "SteamInMicrowave": "microwave"}


def ta(env, task):
    out = {}
    for k in TA[task]:
        v = getattr(env, k, "<unset>")
        if k == "board_contact_positions":
            v = len(v)
        elif isinstance(v, np.ndarray):
            v = [round(float(x), 4) for x in v]
        out[k] = copy.deepcopy(v)
    if task in FX:
        fx = getattr(env, FX[task])
        out["fixture"] = {k: copy.deepcopy(v) for k, v in vars(fx).items() if k in ("_turned_on", "_num_steps_on", "_cooldown")}
    return out


def score(env, task):
    """a scalar of accumulation progress, to choose the node."""
    if task == "WashLettuce": return env.washed_time
    if task == "StirVegetables": return env.success_time
    if task == "ScrubCuttingBoard": return env.board_contact_timer
    if task == "RinseSinkBasin": return sum(env.washed_loc)
    if task == "GetToastedBread": return int(env.toaster_on) + sum(env.toaster._num_steps_on.values())
    if task == "SteamInMicrowave": return int(env.microwave._turned_on)


task, tag = sys.argv[1], sys.argv[2]
row = [r for r in rows() if r["tag"] == tag][0]
env = make_env(task)
load_scene(env, scene_of(row))
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
# pass 1: trajectory of the accumulators
traj = []
for f in range(len(states)):
    set_state(env, states[f]); s = env._check_success()
    traj.append((score(env, task), bool(s)))
first = next((f for f in range(len(states)) if traj[f][0] > 0), None)
succ_f = next((f for f in range(len(states)) if traj[f][1]), None)
if first is None:
    print(json.dumps(dict(task=task, tag=tag, note="accumulator never moved in replay"))); sys.exit()
t_node = int(sys.argv[3]) if len(sys.argv) > 3 else max(16, (first // 16) * 16 + 16)
if len(sys.argv) <= 3 and succ_f is not None and t_node >= succ_f:
    t_node = max(16, ((succ_f - 1) // 16) * 16)
# pass 2: replay to t_node, capture the node as closedloop.py does
load_scene(env, scene_of(row))
for f in range(t_node + 1):
    set_state(env, states[f]); env._check_success()
node = dict(state=env.sim.get_state().flatten().copy(), ps=py_state(env), hidden=hidden_state.capture(env, obs=True))
at_node, succ_node = ta(env, task), bool(env._check_success())
for f in range(t_node + 1, len(states)):  # "candidate rollout": the rest of the episode
    set_state(env, states[f]); env._check_success()
at_end, succ_end = ta(env, task), bool(env._check_success())
res = dict(task=task, tag=tag, n=len(states), first_move=first, replay_success_frame=succ_f, online_steps=row["steps"],
           online_success=row["success"], t_node=t_node, at_node=at_node, success_at_node=succ_node, at_end=at_end,
           success_at_end=succ_end)
# restore exactly as closedloop.py does before every candidate: set_py_state(node.ps) then rollout -> restore_node
after = []
for rep in range(3):
    set_py_state(env, node["ps"]); restore_node(env, node["state"], node["hidden"])
    after.append(dict(attrs=ta(env, task), success=bool(env._check_success())))
res["after_restore_x3"] = after
# a fresh env (another worker / process): load the scene, restore the node
env2 = make_env(task); load_scene(env2, scene_of(row))
set_py_state(env2, node["ps"]); restore_node(env2, node["state"], node["hidden"])
res["fresh_env_restore"] = dict(attrs=ta(env2, task), success=bool(env2._check_success()))
print(json.dumps(res, default=str))
