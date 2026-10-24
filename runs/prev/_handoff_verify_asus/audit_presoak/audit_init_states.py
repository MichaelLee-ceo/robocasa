"""For every PreSoakPan scene s0-19 (scenes_seen5_v2): sink-containment of pan/sponge at the frozen INITIAL state and at
the episode END state, with the library check (OU.obj_inside_of, tolerance 0.05 in dot-product units with unnormalised
region axes -> effective tolerance 0.05/|axis| metres) vs a metric variant (0.05 m and 0.0 m along unit axes).
usage: python -I audit_init_states.py <out.json>"""
import json, os, sys
import numpy as np
import gymnasium as gym

ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import groot_eval  # noqa
from oracle import reset_to, set_state
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T

RUN = f"{ROOT}/runs/20261014_asus_seen5"; SCN = f"{ROOT}/runs/scenes_seen5_v2"
e = gym.make("robocasa/PreSoakPan", split="target", enable_render=False).unwrapped.env


def inside_metric(name, th_m):
    obj = e.objects[name]
    pos = np.array(e.sim.data.body_xpos[e.obj_body_id[obj.name]])
    quat = T.convert_quat(e.sim.data.body_xquat[e.obj_body_id[obj.name]], to="xyzw")
    pts = obj.get_bbox_points(trans=pos, rot=quat)
    for p0, px, py, pz in e.sink.get_int_sites(relative=False).values():
        ok = True
        for a_end in (px, py, pz):
            a = (a_end - p0) / np.linalg.norm(a_end - p0)
            lo, hi = a @ p0 - th_m, a @ a_end + th_m
            if not all(lo <= a @ p <= hi for p in pts):
                ok = False; break
        if ok:
            return True
    return False


def evals():
    return {n: dict(lib=bool(OU.obj_inside_of(e, n, e.sink, partial_check=False)), metric_005=inside_metric(n, 0.05),
                    metric_000=inside_metric(n, 0.0), center=bool(OU.obj_inside_of(e, n, e.sink, partial_check=True)),
                    grip_far=bool(OU.gripper_obj_far(e, n)))
            for n in ("obj1", "obj2")}


res = {}
for s in range(20):
    z = np.load(f"{SCN}/PreSoakPan_s{s}.npz", allow_pickle=True)
    reset_to(e, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); e.unset_ep_meta()
    init = evals()
    set_state(e, np.load(f"{RUN}/PreSoakPan/PreSoakPan_s{s}_single_end.npz", allow_pickle=True)["state"])
    end = evals()
    res[s] = dict(init=init, end=end, water_on_end=bool(e.sink.get_handle_state(env=e)["water_on"]))
    print(s, json.dumps(res[s]), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1])
