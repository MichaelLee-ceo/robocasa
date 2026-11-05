"""PreSoakPan: at t=0 of every frozen scene, evaluate the stage-0 check, the official inside predicate (th=0.05)
and a strict one (th=0), gripper-far, and the margin in metres per sink-region axis. Also replays seen480 step
states 0..16 for s16 / s19 to see whether the sponge moved before the first boundary."""
import os, sys, json
import numpy as np
ROOT = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{ROOT}/runs/scripts")
sys.path.insert(0, f"{ROOT}/third_party/Isaac-GR00T")
import gymnasium as gym
import robocasa  # noqa
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T
from atomic_eval import load_scene
from oracle import set_state
from atomic_steps import TASKS

import robosuite
_orig_make = robosuite.make
def _make(**kw):
    kw["has_offscreen_renderer"] = False; kw["use_camera_obs"] = False
    return _orig_make(**kw)
robosuite.make = _make
from robocasa.utils.env_utils import create_env
TASK = "PreSoakPan"
SEEDS = [int(x) for x in sys.argv[1].split(",")]
PROBE_STEPS = len(sys.argv) > 2
kenv = create_env(env_name=TASK, split="target")


def axis_excess(env, o):
    """for each sink int region: margin (m) per axis = 0.05/|edge|, and how far (m) the worst bbox point lies outside
    the strict region along each axis (positive = outside)."""
    obj = env.objects[o]
    pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
    quat = T.convert_quat(env.sim.data.body_xquat[env.obj_body_id[obj.name]], to="xyzw")
    pts = obj.get_bbox_points(trans=pos, rot=quat)
    out = []
    for name, (p0, px, py, pz) in env.sink.get_int_sites(relative=False).items():
        res = {}
        for ax, p1 in (("x", px), ("y", py), ("z", pz)):
            u = p1 - p0; L = np.linalg.norm(u); uh = u / L
            proj = [float(np.dot(uh, p - p0)) for p in pts]
            exc = max(max(-min(proj), 0.0), max(max(proj) - L, 0.0))
            res[ax] = dict(edge=round(L, 3), margin_m=round(0.05 / L, 3), worst_outside_m=round(exc, 3))
        out.append((name, res))
    return out


rows = []
import time
for s in SEEDS:
    _t = time.time()
    d = "scenes_seen5_v2" if s < 20 else "scenes_seen5_v2_s30"
    path = f"{ROOT}/runs/{d}/PreSoakPan_s{s}.npz"
    load_scene(kenv, path)
    S = {}
    st0 = bool(TASKS["PreSoakPan"][0]["check"](kenv, S))
    r = dict(seed=s, layout=kenv.get_ep_meta().get("layout_id"), stage0_t0=st0, order=S.get("order"))
    for o in ("obj1", "obj2"):
        r[o] = dict(inside_off=bool(OU.obj_inside_of(kenv, o, kenv.sink, partial_check=False)),
                    inside_strict=bool(OU.obj_inside_of(kenv, o, kenv.sink, partial_check=False, th=0.0)),
                    far=bool(OU.gripper_obj_far(kenv, o)))
        if r[o]["inside_off"]:
            r[o]["geom"] = axis_excess(kenv, o)
    r["water_on"] = bool(kenv.sink.get_handle_state(env=kenv)["water_on"])
    r["success_t0"] = bool(kenv._check_success())
    rows.append(r)
    print(json.dumps({k: r[k] for k in ("seed", "layout", "stage0_t0", "order", "water_on")},),
          {o: {k: r[o][k] for k in ("inside_off", "inside_strict", "far")} for o in ("obj1", "obj2")}, flush=True)

print("\nflagged:", [r["seed"] for r in rows if r["stage0_t0"]])
for r in rows:
    for o in ("obj1", "obj2"):
        if r[o].get("geom"):
            print(r["seed"], o, json.dumps(r[o]["geom"]))

# seen480 step states for s16 / s19: sponge displacement and inside flag over steps 0..16
for s in ((16, 19) if PROBE_STEPS else ()):
    load_scene(kenv, f"{ROOT}/runs/scenes_seen5_v2/PreSoakPan_s{s}.npz")
    z = np.load(f"{ROOT}/runs/20261025_seen480/run/PreSoakPan/PreSoakPan_s{s}_single_steps.npz", allow_pickle=True)
    key = [k for k in z.files][0]
    states = z[key] if z[key].ndim == 2 else z["states"]
    p0 = None
    for t in (0, 1, 8, 16, 32):
        set_state(kenv, states[t])
        p = kenv.sim.data.body_xpos[kenv.obj_body_id[kenv.objects["obj2"].name]].copy()
        p0 = p if p0 is None else p0
        g = kenv.sim.data.site_xpos[kenv.robots[0].eef_site_id["right"]]
        print(f"s{s} t={t} sponge_inside={OU.obj_inside_of(kenv, 'obj2', kenv.sink, partial_check=False)} "
              f"strict={OU.obj_inside_of(kenv, 'obj2', kenv.sink, partial_check=False, th=0.0)} "
              f"moved={np.linalg.norm(p - p0):.4f} grip_dist={np.linalg.norm(g - p):.3f} keys={z.files}")
