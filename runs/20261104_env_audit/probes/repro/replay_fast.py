"""Re-execute critic-data rollouts with their RECORDED actions in this process (CPU, Mesa llvmpipe rendering), exactly as
collect.decision_point sets them up, and compare (a) every step's sim state with the saved one, (b) the first policy
request's images with the GPU-rendered ones saved in snap/ (req/...).
usage: replay_check.py <task> <dataset> <tag> <t> [<tag> <t> ...]  (decision points run one after another in ONE env,
so later ones see the worker history of earlier ones)"""
import copy, json, os, sys, time
import numpy as np, gymnasium as gym
R = "/home/michaellee/mclee/robocasa/runs"
sys.path[:0] = [f"{R}/scripts", f"{R}/20261026_imagine", f"{R}/20261026_critic_data"]
import collect as C
from atomic_eval import load_scene, rollout
from atomic_steps import TASKS
import hidden_state
from imagine import py_state, set_py_state

task, ds = sys.argv[1], sys.argv[2]
pts = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(3, len(sys.argv), 2)]
D = f"{R}/20261026_critic_data/data/{task}/{ds}"
rows = {(r["tag"], r["t"]): r for r in map(json.loads, open(f"{D}/results.jsonl"))}
jobs = {(j["tag"], j["t"]): j for j in json.load(open(f"{R}/20261026_critic_data/jobs/{task}.json"))}


class FakeClient:
    def __init__(self, chunks):
        self.chunks, self.i, self.req0 = chunks, 0, None
    def get_action(self, req):
        if self.req0 is None:
            self.req0 = req
        if self.i >= len(self.chunks):
            raise IndexError("replay asked for more chunks than recorded")
        c = self.chunks[self.i]; self.i += 1; RENDER[0] = False
        out = {k: np.asarray(v)[None] for k, v in c.items()}
        out["_seed_echo"] = req.get("_seed")
        return out


base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
RENDER = [True]
_orig_render = base.env.sim.render
def _render(*a, **kw):
    if RENDER[0]:
        return _orig_render(*a, **kw)
    h, w = kw.get("height", 256), kw.get("width", 256)
    z = np.zeros((h, w, 3), np.uint8)
    return (z, np.zeros((h, w), np.float32)) if kw.get("depth") else z
base.env.sim.render = _render
kenv = base.env
steps = TASKS[task]
for tag, t in pts:
    row, j = rows[(tag, t)], jobs.get((tag, t))
    k, L = row["k"], row["L"]
    scene = j["scene"] if j else f"{R}/scenes_seen5_v2/{tag.split('_single')[0]}.npz"
    t0 = time.time()
    load_scene(kenv, scene)
    instr = kenv.get_ep_meta()["lang"]
    ep_states = np.load(f"{C.RUN}/{task}/{tag}_steps.npz")["states"]
    state = ep_states[t]
    S0, kr = C.replay(kenv, ep_states, t, steps)
    if t:
        hidden_state.settle(kenv, ep_states[t - 1])
    H0 = hidden_state.capture(kenv)
    ps = py_state(kenv)
    C.snapshot(kenv)
    res = dict(tag=tag, t=t, k=k)
    for ci in (0, 1):
        z = np.load(f"{D}/roll/{tag}_t{t}_c{ci}_m0.npz")
        akeys = [x for x in z.files if x.startswith("action")]
        chunks = [{a: z[a][q] for a in akeys} for q in range(z[akeys[0]].shape[0])]
        fc, states = FakeClient(chunks), []; RENDER[0] = True
        set_py_state(kenv, ps)
        try:
            rollout(fc, base, kenv, copy.deepcopy(S0), steps, state, L, "task", instr, k0=k, k_stop=k + 1,
                    seed_fn=lambda c: 0, step_states=states, hidden=H0)
            err = None
        except IndexError as e:
            err = str(e)
        saved = z["states"]
        n = min(len(saved), len(states))
        first = next((i for i in range(n) if not np.array_equal(saved[i], states[i])), None)
        dmax = float(np.max(np.abs(np.asarray(states[:n]) - saved[:n])))
        out = dict(steps_saved=len(saved) - 1, steps_replayed=len(states) - 1, first_state_diff=first,
                   max_abs_state_diff=dmax, err=err)
        if ci == 0:
            sn = np.load(f"{D}/snap/{tag}_t{t}.npz")
            imgs = {}
            for key, v in fc.req0.items():
                if key.startswith("video."):
                    g = sn[f"req/{key}"]; c = np.asarray(v)
                    d = np.abs(g.astype(int) - c.astype(int))
                    imgs[key.split(".")[-1]] = dict(frac_px_diff=round(float((d.reshape(-1, 3).max(1) > 0).mean()), 4),
                                                    mean_abs=round(float(d.mean()), 3), max_abs=int(d.max()),
                                                    frac_px_gt8=round(float((d.reshape(-1, 3).max(1) > 8).mean()), 4))
                elif key.startswith("state."):
                    imgs[key] = bool(np.array_equal(sn[f"req/{key}"], np.asarray(v)))
            out["first_request_cpu_vs_gpu"] = imgs
        res[f"c{ci}"] = out
    res["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
