"""Which controller attribute left over by a previous rollout changes a plain rollout from load_scene? (2026-10-30)
Fresh env: load_scene, capture H1, plain rollout (reference). Then another rollout, load_scene, capture H2. For each group
of differing attributes, H2 with that group set back to H1's values is restored and the plain rollout re-run: if it then
equals the reference bit for bit, the group is the cause. Also reports the size of each attribute difference and of the
state difference after 1 and 16 steps without any reset.
usage: python -I reset_culprit.py <tag> <port> [steps=32]"""
import copy, json, os, sys
import numpy as np
import gymnasium as gym
R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # runs/
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/scripts", f"{R}/20261026_imagine", f"{R}/20261026_critic_data", HERE]
from atomic_eval import load_scene, rollout  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from groot_eval import Client  # noqa: E402
from collect import Recorder  # noqa: E402
from closedloop import scene, seed  # noqa: E402
from preflight import diff  # noqa: E402
import hidden_state  # noqa: E402

tag, port = sys.argv[1], int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
task = tag.split("_s")[0]; steps = TASKS[task]
base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped; kenv = base.env
client = Client(port=port, timeout_ms=600000)
rec0 = Recorder(client)
def run(hidden=None, first=None):
    s0 = load_scene(kenv, scene(task, tag)); instr = kenv.get_ep_meta()["lang"]
    st = []
    rollout(Recorder(client, first=first), base, kenv, {}, steps, s0, n, "task", instr, seed_fn=lambda ci: seed(tag, 1, 0, ci),
            step_states=st, hidden=hidden)
    return st
s0 = load_scene(kenv, scene(task, tag)); instr = kenv.get_ep_meta()["lang"]
H1 = hidden_state.capture(kenv)
ref = []
rollout(rec0, base, kenv, {}, steps, s0, n, "task", instr, seed_fn=lambda ci: seed(tag, 1, 0, ci), step_states=ref)
chunk0 = rec0.chunks[0]
rollout(Recorder(client), base, kenv, {}, steps, s0, 64, "task", instr, seed_fn=lambda ci: seed(tag, 2, 0, ci))
load_scene(kenv, scene(task, tag)); H2 = hidden_state.capture(kenv)
d = [p for p in diff(H1, H2) if "/buffers/" not in p]
def val(h, p):
    for k in [x for x in p.replace("[", "/").replace("]", "").split("/") if x]:
        h = h[int(k)] if isinstance(h, list) else h[k]
    return h
sizes = {}
for p in d:
    try:
        a, b = np.asarray(val(H1, p), float), np.asarray(val(H2, p), float)
        sizes[p] = float(np.max(np.abs(a - b))) if a.shape == b.shape else "shape"
    except Exception as e:
        sizes[p] = repr(val(H1, p))[:40] + " vs " + repr(val(H2, p))[:40]
print(json.dumps(dict(tag=tag, differing=sizes)), flush=True)
def same(st):
    return next((i for i, (a, b) in enumerate(zip(ref, st)) if not np.array_equal(a, b)), None)
# without any fix: the plain rollout after the previous one, same first chunk (isolates physics from the policy)
st = run(first=chunk0)
dd = [float(np.max(np.abs(ref[i] - st[i]))) for i in (1, 16, min(n, len(st) - 1))]
print(json.dumps(dict(case="stale (no fix)", first_diff=same(st), max_state_diff_at_1_16_end=dd)), flush=True)
groups = sorted({"/".join(p.split("/")[:4]) for p in d})
for g in groups + ["all"]:
    H = copy.deepcopy(H2)
    for p in d:
        if g == "all" or p.startswith(g + "/") or p == g:
            keys = [x for x in p.replace("[", "/").replace("]", "").split("/") if x]
            o = H
            for k in keys[:-1]:
                o = o[int(k)] if isinstance(o, list) else o[k]
            o[keys[-1]] = copy.deepcopy(val(H1, p))
    st = run(hidden=H, first=chunk0)
    print(json.dumps(dict(case=f"stale, {g} set to fresh", first_diff=same(st))), flush=True)
