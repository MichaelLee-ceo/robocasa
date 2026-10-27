"""Does a rollout started from a saved state depend on what ran before it? (hidden_state.py; found 2026-10-27)

At a KettleBoiling decision point, run a fixed 16-step chunk X (actions taken from the collected data), then 64 other
steps that leave the controllers somewhere else, then restore the state and run X again. Compares the two runs of X:
  1. set_state only (as collected so far);
  2. set_state + restore(capture()) of the hidden state captured at the first start (exact);
  3. set_state + restore() of a settle()d hidden state (the canonical one used at decision points);
  4. fidelity of settle(): continuous execution X then Z vs. restoring the state after X with settle() and running Z.
usage (groot env): python check_restore.py [tag] [t]"""
import glob, json, os, sys
import numpy as np
import gymnasium as gym

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from atomic_eval import load_scene  # noqa: E402
from oracle import set_state  # noqa: E402
import hidden_state as HS  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "KettleBoiling_s0_single"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 160
task = TAG.split("_")[0]
scene = next(j["scene"] for j in json.load(open(f"{HERE}/jobs.json" if task == "KettleBoiling" else f"{HERE}/jobs_{task}.json")) if j["tag"] == TAG)
run_dir = "run" if task == "KettleBoiling" else f"run_{task}"
ep = np.load(f"{HERE}/../20261025_seen480/run/{task}/{TAG}_steps.npz")["states"]


def actions(c, m, chunks):
    z = np.load(f"{HERE}/{run_dir}/roll/{TAG}_t{T}_c{c}_m{m}.npz")
    keys = [k for k in z.files if k.startswith("action.")]
    return [{k: z[k][ci, i] for k in keys} for ci in chunks for i in range(16)]


base = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped
kenv = base.env
load_scene(kenv, scene)
for i in range(T + 1):  # as collect.py's replay()
    set_state(kenv, ep[i])
S = ep[T]
X, Y, Z = actions(0, 0, [0]), actions(3, 1, [0, 1, 2, 3]), actions(0, 0, [1])
ARM, FING = slice(5, 12), slice(12, 14)  # flattened state = [time, qpos...]: qpos 4-10 arm, 11-12 fingers


def run(acts):
    out = []
    for a in acts:
        base.step(a)
        out.append(kenv.sim.get_state().flatten().copy())
    return np.array(out)


def report(name, a, b):
    d = np.abs(a[:, 1:] - b[:, 1:])
    arm = np.degrees(np.abs(a[-1, ARM] - b[-1, ARM]).max()); fing = 1000 * np.abs(a[-1, FING] - b[-1, FING]).max()
    print(f"{name:58s} max |state diff| {d.max():.2e}  first differing step {int(np.argmax(d.max(1) > 0)) + 1 if d.max() > 0 else '-'}"
          f"  after 16 steps: arm {arm:.3f} deg, fingers {fing:.3f} mm")


# 1. set_state only
set_state(kenv, S); a = run(X); run(Y)
set_state(kenv, S); b = run(X)
report("1. set_state only (as collected so far)", a, b)
# 2. exact capture / restore
set_state(kenv, S); H = HS.capture(kenv); a = run(X); run(Y); run(X)
set_state(kenv, S); HS.restore(kenv, H); b = run(X)
report("2. set_state + restore(capture()) at the start", a, b)
# 3. settled hidden state, restored before each run
set_state(kenv, S); HS.settle(kenv, ep[T - 1]); Hs = HS.capture(kenv)
a = run(X); run(Y)
set_state(kenv, S); HS.restore(kenv, Hs); b = run(X)
report("3. set_state + restore(settled) at the start", a, b)
# 4. fidelity: continuous X then Z vs. restoring after X
set_state(kenv, S); HS.restore(kenv, Hs); run(X); S16, P15, H16 = kenv.sim.get_state().flatten().copy(), None, HS.capture(kenv)
cont = run(Z)
set_state(kenv, S16); HS.restore(kenv, H16); b = run(Z)
report("4a. continuous vs restore(capture()) after X (exact)", cont, b)
run(Y)
set_state(kenv, S16); HS.settle(kenv); b = run(Z)
report("4b. continuous vs settle() after X (decision-point approx.)", cont, b)
