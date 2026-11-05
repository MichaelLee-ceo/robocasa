"""CPU-only reproducibility probes on a t = 0 critic decision point, with RECORDED actions (no policy):
  1. fresh env: load_scene, capture H1, render each camera K times (render determinism), run candidate c0's recorded
     actions -> states A (compared with the saved GPU-run states);
  2. same env: run c1's recorded actions (a 'previous job'), load_scene again, capture H2, run c0 again -> states B.
     Reports the hidden-state paths that differ between H1 and H2 and where / how much B departs from A (worker history).
Writes <out>.npz (states A, first images) for comparison across processes / env settings.
usage: physics_check.py <task> <tag> <out prefix> [K=10]"""
import copy, json, os, sys, time
import numpy as np, gymnasium as gym
R = "/home/michaellee/mclee/robocasa/runs"
sys.path[:0] = [f"{R}/scripts", f"{R}/20261026_imagine", f"{R}/20261026_critic_data", f"{R}/20261029_closedloop"]
import collect as C
from atomic_eval import load_scene, rollout
from atomic_steps import TASKS
import hidden_state
from imagine import py_state, set_py_state
from preflight import diff

task, tag, outp = sys.argv[1], sys.argv[2], sys.argv[3]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
D = f"{R}/20261026_critic_data/data/{task}/coarse"
row = next(r for r in map(json.loads, open(f"{D}/results.jsonl")) if r["tag"] == tag and r["t"] == 0)
job = next(j for j in json.load(open(f"{R}/20261026_critic_data/jobs/{task}.json")) if j["tag"] == tag and j["t"] == 0)
k, L = row["k"], row["L"]


class FakeClient:
    def __init__(self, chunks): self.chunks, self.i, self.req0 = chunks, 0, None
    def get_action(self, req):
        if self.req0 is None: self.req0 = req
        if self.i >= len(self.chunks): raise IndexError("more chunks than recorded")
        c = self.chunks[self.i]; self.i += 1; RENDER[0] = False
        out = {kk: np.asarray(v)[None] for kk, v in c.items()}; out["_seed_echo"] = req.get("_seed"); return out


def chunks_of(ci):
    z = np.load(f"{D}/roll/{tag}_t0_c{ci}_m0.npz")
    ak = [x for x in z.files if x.startswith("action")]
    return [{a: z[a][q] for a in ak} for q in range(z[ak[0]].shape[0])], z["states"]


base = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped
kenv = base.env; steps = TASKS[task]
RENDER = [True]
def patch():
    real = kenv.sim.render
    def _render(*a, **kw):
        if RENDER[0]:
            return real(*a, **kw)
        h, w = kw.get("height", 256), kw.get("width", 256)
        z = np.zeros((h, w, 3), np.uint8)
        return (z, np.zeros((h, w), np.float32)) if kw.get("depth") else z
    kenv.sim.render = _render
    return real
def point():
    load_scene(kenv, job["scene"]); instr = kenv.get_ep_meta()["lang"]
    point.real = patch()
    ep = np.load(f"{C.RUN}/{task}/{tag}_steps.npz")["states"]
    S0, _ = C.replay(kenv, ep, 0, steps)
    H0 = hidden_state.capture(kenv); ps = py_state(kenv)
    return ep[0], S0, H0, ps, instr
def run(ci, state, S0, H0, ps, instr):
    ch, saved = chunks_of(ci); fc, st = FakeClient(ch), []; RENDER[0] = True
    set_py_state(kenv, ps)
    try:
        rollout(fc, base, kenv, copy.deepcopy(S0), steps, state, L, "task", instr, k0=k, k_stop=k + 1, seed_fn=lambda c: 0,
                step_states=st, hidden=H0)
    except IndexError:
        pass
    return np.asarray(st), saved, fc
def cmp(a, b):
    n = min(len(a), len(b)); d = np.abs(a[:n] - b[:n]).max(1)
    first = next((i for i in range(n) if d[i] > 0), None)
    return dict(n=n, len=[len(a), len(b)], first_diff_step=first,
                maxdiff_at={s: float(d[min(s, n - 1)]) for s in (0, 1, 16, 64, 128, 256, n - 1)})
res = dict(task=task, tag=tag, env={v: os.environ.get(v) for v in ("LP_NUM_THREADS", "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "OMP_NUM_THREADS")})
state, S0, H1, ps, instr = point()
imgs = {}
t0 = time.time()
for cam in ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"):
    ims = [point.real(camera_name=cam, width=256, height=256) for _ in range(K)]
    res.setdefault("cpu_render_distinct_of_K", {})[cam] = len({x.tobytes() for x in ims}); imgs[cam] = ims[0]
res["cpu_sec_per_render"] = round((time.time() - t0) / (3 * K), 4)
A, savedA, fcA = run(0, state, S0, H1, ps, instr)
res["A_vs_saved_gpu_run"] = cmp(A, savedA)
_ = run(1, state, S0, H1, ps, instr)  # a different rollout in the same env = the previous job of a worker
state2, S02, H2, ps2, _ = point()
res["H1_vs_H2_after_history"] = [p for p in diff(H1, H2) if "/buffers/" not in p][:20]
res["H1_vs_H2_buffers_only"] = len([p for p in diff(H1, H2) if "/buffers/" in p])
B, _, fcB = run(0, state2, S02, H2, ps2, instr)
res["B_vs_A"] = cmp(B, A)
res["B_req0_eq_A_req0_state_keys"] = all(np.array_equal(fcA.req0[x], fcB.req0[x]) for x in fcA.req0 if x.startswith("state."))
np.savez_compressed(outp + ".npz", A=A, **{f"img/{c}": v for c, v in imgs.items()},
                    **{f"req0/{kk}": np.asarray(v) for kk, v in fcA.req0.items() if kk.startswith(("video", "state"))})
print(json.dumps(res), flush=True)
