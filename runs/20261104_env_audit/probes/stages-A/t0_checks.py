"""every stage check (S={}) at the initial state of every frozen scene"""
import sys, time
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/stages-A")
from common import *
from atomic_steps import TASKS
task = sys.argv[1]
env = make_env(task)
for s in range(30):
    t0 = time.time()
    try:
        load(env, task, s)
    except Exception as e:
        print(task, s, "ERR", repr(e)[:200]); continue
    res = [bool(st["check"](env, {})) for st in TASKS[task]]
    extra = {}
    if task == "PreSoakPan":
        extra = {o: (bool(OU.obj_inside_of(env, o, env.sink, partial_check=False)), bool(OU.gripper_obj_far(env, o))) for o in ("obj1", "obj2")}
    if task == "KettleBoiling":
        st = env.stove
        extra = dict(sites=[k for k, v in st.burner_sites.items() if v is not None], knobs=list(st.get_knobs_state(env=env).keys()))
        import numpy as np
        xy = {k: np.array(env.sim.data.get_site_xpos(v.get("name")))[:2] for k, v in st.burner_sites.items() if v is not None}
        ks = list(xy)
        extra["min_pair_dist"] = round(min(float(np.linalg.norm(xy[a] - xy[b])) for i, a in enumerate(ks) for b in ks[i + 1:]), 3)
    print(task, s, res, extra, round(time.time() - t0, 1), flush=True)
