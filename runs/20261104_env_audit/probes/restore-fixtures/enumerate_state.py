"""For one task: which python attributes (env, fixtures, objects) change over a saved baseline episode (replayed as
stage_trace does: set_state + _check_success per frame) and over live random steps; and whether hidden_state covers them.
usage: python enumerate_state.py TASK [TAG]"""
import sys, json, time
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
from robocasa.models.fixtures.fixture import Fixture

task = sys.argv[1]
R = [r for r in rows() if r["task"] == task]
if len(sys.argv) > 2:
    row = [r for r in R if r["tag"] == sys.argv[2]][0]
else:
    succ = [r for r in R if r["success"]]
    row = succ[0] if succ else R[0]
tag = row["tag"]
t0 = time.time()
env = make_env(task)
start = load_scene(env, scene_of(row))
states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
A0 = snap(env)
fx_upd = sorted(n + ":" + type(fx).__name__ for n, fx in env.fixtures.items() if type(fx).update_state is not Fixture.update_state)
changed = set()
prev = A0
for f in range(len(states)):
    set_state(env, states[f])
    env._check_success()
    if f % 8 == 0 or f == len(states) - 1:
        cur = snap(env)
        changed |= set(diff(prev, cur))
        prev = cur
ps = py_state(env)
out = dict(task=task, tag=tag, success=row["success"], n=len(states), fixtures_with_update_state=fx_upd, replay_changed=[])
for (owner, k) in sorted(changed):
    cov = "UNCOVERED"
    if owner.startswith("fx:"):
        n = owner.split(":")[1]
        if n in ps and k in ps[n]:
            cov = "py_state"
        elif n in ps:
            cov = "UNCOVERED (fixture has update_state, value not _plain)"
        else:
            cov = "UNCOVERED (fixture class has no update_state override)"
    elif owner == "env" and k in ("timestep", "cur_time"):
        cov = "capture"
    v = getattr(env, k, None) if owner == "env" else None
    out["replay_changed"].append([owner, k, cov, repr(v)[:80] if owner == "env" else ""])
# live stepping from the scene start: env attributes that step() changes
load_scene(env, scene_of(row))
B0 = snap(env)
rng = np.random.default_rng(0)
low, high = env.action_spec
for i in range(40):
    a = rng.uniform(low, high) * 0.3
    env.step(a)
B1 = snap(env)
out["live_changed"] = [[o, k] for (o, k) in diff(B0, B1)]
out["wall_s"] = round(time.time() - t0, 1)
json.dump(out, open(f"enum_{task}.json", "w"), indent=1)
print("done", task, out["wall_s"])
