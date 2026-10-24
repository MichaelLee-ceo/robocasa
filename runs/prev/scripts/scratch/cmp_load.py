import sys, os, json
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
from atomic_eval import load_scene
from oracle import set_state
base = gym.make("robocasa/StackBowlsCabinet", split="target", enable_render=True).unwrapped; kenv = base.env
def snap():
    o = base.get_observation(kenv._get_observations(force_update=True)); return {k: np.asarray(v, dtype=float) for k, v in o.items() if not isinstance(v, str)}
def roll(n=48):
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        a = {k: rng.uniform(-.3, .3, size=sp.shape).astype(sp.dtype) if hasattr(sp, "shape") and sp.shape else sp.sample() for k, sp in base.action_space.spaces.items()}
        o, *_ = base.step(a); out.append(np.concatenate([np.ravel(o[k]) for k in sorted(o) if k.startswith("state.")]))
    return np.array(out)
base.reset(seed=0); A = snap(); sA = kenv.sim.get_state().flatten(); RA = roll()
x = load_scene(kenv, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scenes_seen5/StackBowlsCabinet_s0.npz"); B = snap(); sB = kenv.sim.get_state().flatten(); RB = roll()
print("state vec equal shape", sA.shape == sB.shape, "max diff", float(np.abs(sA - sB).max()) if sA.shape == sB.shape else None)
for k in A:
    if A[k].shape == B[k].shape:
        d = float(np.abs(A[k] - B[k]).max())
        if d > 1e-4: print("obs differs", k, round(d, 4))
    else: print("shape differs", k)
print("rollout state divergence after 48 random steps", float(np.abs(RA - RB).max()))
print("action space", {k: getattr(v, "shape", None) for k, v in base.action_space.spaces.items()})
print("ctrl", type(kenv.robots[0].composite_controller).__name__)
