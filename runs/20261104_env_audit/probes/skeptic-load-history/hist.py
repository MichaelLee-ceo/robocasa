"""Does the hidden state after load_scene, and the stepped state after it, depend on what the process loaded before?
A: fresh env, load s0.  B: load s0 again right after A.  C: load s1, then s0.  D: load s0, step 64 (a rollout), load s0.
For each: anchor (init_robot_base_ori/pos_anchor from the preceding random reset), sim state after reset_from_xml_string's
settle loop, capture() (non-buffer), then restore_node + forced obs + 32 fixed steps (arm mode and base mode mixed)."""
import sys, json, time
HERE = sys.argv[1]; sys.path.insert(0, HERE)
from harness import *
import atomic_eval
from atomic_eval import load_scene, restore_node
import hidden_state

def flat(h, pre=""):
    out = {}
    if isinstance(h, dict):
        for k, x in h.items():
            out.update(flat(x, f"{pre}/{k}"))
    elif isinstance(h, (list, tuple)) and h and isinstance(h[0], dict):
        for i, x in enumerate(h):
            out.update(flat(x, f"{pre}[{i}]"))
    else:
        out[pre] = h
    return out

def cmp(a, b):
    fa, fb = flat(a), flat(b); bad = {}
    for k in sorted(set(fa) | set(fb)):
        if "/buffers/" in k or k.startswith("/obs_clock"):
            continue
        x, y = fa.get(k), fb.get(k)
        try:
            same = np.array_equal(np.asarray(x), np.asarray(y))
        except Exception:
            same = repr(x) == repr(y)
        if not same:
            try:
                bad[k] = float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))))
            except Exception:
                bad[k] = "?"
    return bad

task = sys.argv[2]
env = make_env(task)
post_xml = {}
orig = env.reset_from_xml_string
def wrapped(x):
    r = orig(x); post_xml["state"] = env.sim.get_state().flatten().copy(); return r
env.reset_from_xml_string = wrapped
acts = [action_vec(env, *a) for a in rand_actions(32, 7, p_base=0.4)]

def trial(path, extra=None):
    st = load_scene(env, path)
    rec = dict(anchor_ori=np.array(env.init_robot_base_ori_anchor, float), anchor_pos=np.array(env.init_robot_base_pos_anchor, float),
               post_xml=post_xml["state"].copy(), H=hidden_state.capture(env))
    restore_node(env, st, None)
    env._get_observations(force_update=True)
    states = []
    for a in acts:
        env.step(a); states.append(env.sim.get_state().flatten().copy())
    rec["steps"] = states
    return rec

s0, s1 = scene_path(task, 0), scene_path(task, 1)
t0 = time.time()
A = trial(s0)
B = trial(s0)
load_scene(env, s1)
C = trial(s0)
def summ(X, Y):
    fd = next((i for i, (a, b) in enumerate(zip(X["steps"], Y["steps"])) if not np.array_equal(a, b)), None)
    return dict(anchor_ori_diff=float(np.max(np.abs(X["anchor_ori"] - Y["anchor_ori"]))),
                anchor_pos_diff=float(np.max(np.abs(X["anchor_pos"] - Y["anchor_pos"]))),
                post_xml_settle_diff=float(np.max(np.abs(X["post_xml"] - Y["post_xml"]))) if X["post_xml"].shape == Y["post_xml"].shape else "shape",
                hidden=cmp(X["H"], Y["H"]),
                first_step_diff=fd,
                state_diff_at_1_8_32=[float(np.max(np.abs(X["steps"][i] - Y["steps"][i]))) for i in (0, 7, 31)])
print("RESULT", json.dumps(dict(task=task, anchor_ori=[A["anchor_ori"].tolist(), B["anchor_ori"].tolist(), C["anchor_ori"].tolist()],
                                AB=summ(A, B), AC=summ(A, C), BC=summ(B, C), secs=round(time.time() - t0)), default=str), flush=True)
