"""Hidden state right after load_scene: (a) is it the same when the same scene is loaded again later in the process
(the main run loaded many scenes per process; imagine / closedloop capture it as the root node)? (b) how stale is the
OSC origin (base frame used by the first set_goal) relative to the base pose of the loaded state?"""
import sys, json
sys.path.insert(0, sys.argv[1])
from harness import *
from atomic_eval import load_scene
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
    fa, fb = flat(a), flat(b)
    bad = []
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k, "MISSING"), fb.get(k, "MISSING")
        try:
            same = np.array_equal(np.asarray(x, dtype=object) if not isinstance(x, np.ndarray) else x,
                                  np.asarray(y, dtype=object) if not isinstance(y, np.ndarray) else y)
        except Exception:
            same = repr(x) == repr(y)
        if not same:
            try:
                d = float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))))
            except Exception:
                d = None
            bad.append((k, d))
    return bad

def stale(env):
    r = env.robots[0]; cc = r.composite_controller; c = cc.part_controllers["right"]
    p, R = cc.get_controller_base_pose("right")
    yaw = lambda M: float(np.arctan2(M[1, 0], M[0, 0]))
    # where the first arm-mode set_goal (zero delta) would send the eef: goal from the stale origin, executed with the fresh one
    g = c.world_to_origin_frame(np.array(c.ref_pos))
    des = p + R @ g
    return dict(origin_pos_err_mm=round(1000 * float(np.linalg.norm(np.asarray(c.origin_pos) - p)), 2),
                origin_yaw_err_deg=round(float(np.degrees(yaw(np.asarray(c.origin_ori)) - yaw(R))), 3),
                first_goal_offset_mm=round(1000 * float(np.linalg.norm(des - c.ref_pos)), 2),
                base_init_yaw_deg=round(float(np.degrees(yaw(cc.part_controllers["base"].init_ori))), 3),
                base_yaw_now_deg=round(float(np.degrees(yaw(R))), 3))

task = sys.argv[2]
env = make_env(task)
s0, s1 = scene_path(task, 0), scene_path(task, 1)
load_scene(env, s0); H1 = hidden_state.capture(env); st1 = stale(env); x1 = env.sim.get_state().flatten()
load_scene(env, s1); load_scene(env, s1)
load_scene(env, s0); H2 = hidden_state.capture(env); st2 = stale(env); x2 = env.sim.get_state().flatten()
bad = cmp(H1, H2)
print("RESULT", json.dumps(dict(task=task, same_sim=bool(np.array_equal(x1, x2)), hidden_diffs=[(k, d) for k, d in bad if "obs_clock" not in k][:12],
                                n_diffs=len(bad), stale_first=st1, stale_second=st2), default=str))
