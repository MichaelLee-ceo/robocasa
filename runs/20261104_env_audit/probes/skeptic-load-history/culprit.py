"""Is the stepped divergence after a second load_scene of the same scene caused by the hidden (controller) state only?
A: fresh env, load s0, capture H_A, step N.  B: load s0 again, step N (stale).  B': load s0 again, restore(H_A), step N.
Also compares every mjModel array between the two loads."""
import sys, json, copy
HERE = sys.argv[1]; sys.path.insert(0, HERE)
from harness import *
from atomic_eval import load_scene, restore_node
import hidden_state, mujoco
task = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 32
s_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
env = make_env(task)
acts = [action_vec(env, *a) for a in rand_actions(N, 7, p_base=0.4)]
def model_arrays():
    mm = env.sim.model._model; out = {}
    for k in dir(mm):
        if k.startswith("_"): continue
        try: v = getattr(mm, k)
        except Exception: continue
        if isinstance(v, np.ndarray) and v.dtype != object: out[k] = v.copy()
    return out
def run(path, H=None):
    st = load_scene(env, path)
    Hc = hidden_state.capture(env); M = model_arrays()
    if H is not None: hidden_state.restore(env, H)
    restore_node(env, st, None); env._get_observations(force_update=True)
    out = []
    for a in acts:
        env.step(a); out.append(env.sim.get_state().flatten().copy())
    return Hc, M, out
p = scene_path(task, s_idx)
HA, MA, A = run(p)
HB, MB, B = run(p)
_, MC, C = run(p, H=HA)
fd = lambda X, Y: next((i for i, (a, b) in enumerate(zip(X, Y)) if not np.array_equal(a, b)), None)
md = [k for k in MA if k in MB and (MA[k].shape != MB[k].shape or not np.array_equal(MA[k], MB[k]))]
print("RESULT", json.dumps(dict(task=task, scene=s_idx, model_arrays_differ_AB=md,
      stale=dict(first_diff=fd(A, B), maxdiff_1_8_N=[float(np.max(np.abs(A[i] - B[i]))) for i in (0, 7, N - 1)]),
      restoredHA=dict(first_diff=fd(A, C), maxdiff_1_8_N=[float(np.max(np.abs(A[i] - C[i]))) for i in (0, 7, N - 1)]))), flush=True)
