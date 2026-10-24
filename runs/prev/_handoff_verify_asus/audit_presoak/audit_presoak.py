"""Offline audit of PreSoakPan end states (ASUS seen5 run). Reloads frozen scene + end npz and evaluates
faucet / sink-containment checks in detail. No rendering, no policy.
usage: python -I audit_presoak.py <out.json>"""
import json, os, sys
import numpy as np
import gymnasium as gym

ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import groot_eval  # noqa  (registers robocasa/* envs)
from oracle import reset_to, set_state
from atomic_steps import TASKS
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array
from robocasa.models.fixtures.sink import Sink

RUN = f"{ROOT}/runs/20261014_asus_seen5"
SCN = f"{ROOT}/runs/scenes_seen5_v2"
STEPS = TASKS["PreSoakPan"]
out_path = sys.argv[1]

env = gym.make("robocasa/PreSoakPan", split="target", enable_render=False).unwrapped
e = env.env


def load(seed, end=True):
    z = np.load(f"{SCN}/PreSoakPan_s{seed}.npz", allow_pickle=True)
    reset_to(e, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); e.unset_ep_meta()
    init_state = z["state"].copy()
    if end:
        en = np.load(f"{RUN}/PreSoakPan/PreSoakPan_s{seed}_single_end.npz", allow_pickle=True)
        set_state(e, en["state"])
    return init_state


def r(x, n=4):
    return np.round(np.asarray(x, dtype=float), n).tolist()


def faucet(seed):
    m, d = e.sim.model, e.sim.data
    sink = e.sink
    pre = sink.naming_prefix
    out = dict(seed=seed, sink_name=sink.name)
    out["all_sinks"] = [f.name for f in e.fixtures.values() if isinstance(f, Sink)]
    jn = f"{pre}handle_joint"
    jid = m.joint_name2id(jn)
    adr = m.jnt_qposadr[jid]
    out["handle_joint_id"] = int(jid); out["handle_qposadr"] = int(adr)
    out["get_joint_qpos_addr"] = int(m.get_joint_qpos_addr(jn))
    out["qpos_by_adr"] = float(d.qpos[adr]); out["qpos_by_id(wrong-if-used)"] = float(d.qpos[jid])
    out["jnt_range_model"] = r(m.jnt_range[jid]); out["range_xml_elem"] = sink.handle_joint.get("range")
    hs = sink.get_handle_state(env=e)
    out["handle_state"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in hs.items()}
    sid = m.site_name2id(f"{pre}water")
    out["water_site_rgba_alpha_after_reload"] = float(m.site_rgba[sid][3])
    out["water_site_size0"] = float(m.site_size[sid][0])
    # every sink-like joint, in case another sink/faucet exists
    out["sink_joints"] = {m.joint_id2name(j): float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
                          if (m.joint_id2name(j) or "").startswith(pre)}
    c = dict(water_on=bool(hs.get("water_on")),
             pan_in_sink=bool(OU.obj_inside_of(e, "obj1", e.sink, partial_check=False)),
             sponge_in_sink=bool(OU.obj_inside_of(e, "obj2", e.sink, partial_check=False)),
             grip_far_pan=bool(OU.gripper_obj_far(e, "obj1")), grip_far_sponge=bool(OU.gripper_obj_far(e, "obj2")))
    c["_check_success"] = bool(e._check_success())
    c["tracker_steps"] = [bool(s["check"](e, {})) for s in STEPS]
    out["success_terms"] = c
    return out


def regions_world(fx):
    res = {}
    for name, (p0, px, py, pz) in fx.get_int_sites(relative=False).items():
        u, v, w = px - p0, py - p0, pz - p0
        res[name] = dict(p0=p0, px=px, py=py, pz=pz, u=u, v=v, w=w,
                         len=[np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(w)])
    return res


def contain_detail(obj_name, th=0.05):
    """Re-implements OU.obj_inside_of (object_utils.py:14-64) and reports, per region and per bbox corner,
    the signed margin in metres along each region axis (negative = outside the 0.05-dot-product tolerance)."""
    obj = e.objects[obj_name]
    pos = np.array(e.sim.data.body_xpos[e.obj_body_id[obj.name]])
    quat = T.convert_quat(e.sim.data.body_xquat[e.obj_body_id[obj.name]], to="xyzw")
    pts = obj.get_bbox_points(trans=pos, rot=quat)
    R = T.quat2mat(quat)
    loc = [R.T @ (p - pos) for p in pts]
    out = dict(obj=obj_name, obj_pos=r(pos), corners_world=[r(p) for p in pts], corners_objlocal=[r(p) for p in loc],
               lib_result=bool(OU.obj_inside_of(e, obj_name, e.sink, partial_check=False)), regions={})
    for name, g in regions_world(e.sink).items():
        rows, inside = [], True
        for i, p in enumerate(pts):
            mm = []
            for ax, a, hi in (("u", g["u"], g["px"]), ("v", g["v"], g["py"]), ("w", g["w"], g["pz"])):
                L = np.linalg.norm(a); ah = a / L
                s, lo_, hi_ = ah @ p, ah @ g["p0"], ah @ hi
                tol = th / L  # effective tolerance in metres (dot product with unnormalised axis)
                mm.append(dict(axis=ax, coord=round(float(s - lo_), 4), region_len=round(float(L), 4),
                               tol_m=round(float(tol), 4),
                               margin_lo=round(float(s - (lo_ - tol)), 4), margin_hi=round(float((hi_ + tol) - s), 4),
                               margin_lo_strict=round(float(s - lo_), 4), margin_hi_strict=round(float(hi_ - s), 4)))
            ok = all(x["margin_lo"] >= 0 and x["margin_hi"] >= 0 for x in mm)
            inside &= ok
            rows.append(dict(corner=i, ok=ok, axes=mm))
        out["regions"][name] = dict(p0=r(g["p0"]), px=r(g["px"]), py=r(g["py"]), pz=r(g["pz"]), len=r(g["len"]),
                                    tol_m=r([th / L for L in g["len"]]), inside=bool(inside), corners=rows)
    return out


def mesh_profile(obj_name, nb=10):
    """Vertex distribution of the object's meshes in its body frame: width across the short horizontal axis
    in bins along the long horizontal axis (narrow end bins = handle)."""
    m, d = e.sim.model, e.sim.data
    bid = e.obj_body_id[obj_name]
    pos, R = d.body_xpos[bid], d.body_xmat[bid].reshape(3, 3)
    sub = {bid}
    for b in range(m.nbody):
        bb = b
        while bb != 0 and bb not in sub:
            bb = m.body_parentid[bb]
        if bb in sub:
            sub.add(b)
    V = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in sub and m.geom_type[g] == 7 and m.geom_dataid[g] >= 0:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            vw = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            V.append((vw - pos) @ R)
    if not V:
        return None
    V = np.concatenate(V)
    ext = V.max(0) - V.min(0)
    la = int(np.argmax(ext[:2])); sa = 1 - la
    edges = np.linspace(V[:, la].min(), V[:, la].max(), nb + 1)
    prof = []
    for i in range(nb):
        sel = (V[:, la] >= edges[i]) & (V[:, la] <= edges[i + 1])
        w = float(np.ptp(V[sel, sa])) if sel.sum() > 1 else 0.0
        prof.append(dict(bin=[round(float(edges[i]), 3), round(float(edges[i + 1]), 3)], width=round(w, 3)))
    return dict(long_axis="xyz"[la], vert_min=r(V.min(0)), vert_max=r(V.max(0)), profile=prof)


res = {}
# Q1: faucet for s12, s13, s19 (and every seed for context)
res["faucet"] = {}
for s in range(25):
    try:
        load(s)
        res["faucet"][s] = faucet(s)
    except Exception as ex:
        res["faucet"][s] = dict(error=repr(ex))
    f = res["faucet"][s]
    print("faucet", s, f.get("qpos_by_adr"), f.get("handle_state", {}).get("water_on"), f.get("success_terms"), flush=True)

# Q2: s13 pan
load(13)
res["s13_pan"] = contain_detail("obj1")
res["s13_pan"]["mesh_profile"] = mesh_profile("obj1")
res["s13_sponge"] = contain_detail("obj2")
res["s13_basin_loc_full"] = e.sink.get_obj_basin_loc(e, "obj1", partial_check=False) if "basin_left" in e.sink._regions else None
res["s13_basin_loc_center"] = e.sink.get_obj_basin_loc(e, "obj1", partial_check=True) if "basin_left" in e.sink._regions else None
print("s13 pan", res["s13_pan"]["lib_result"], {k: v["inside"] for k, v in res["s13_pan"]["regions"].items()}, flush=True)

# Q3: s19 sponge moved onto counter next to basin
init = load(19, end=False)
sp_bid = e.obj_body_id["obj2"]
sp_init_pos = np.array(e.sim.data.body_xpos[sp_bid]); sp_init_quat = np.array(e.sim.data.body_xquat[sp_bid])
load(19)
end_state = e.sim.get_state().flatten().copy()
jname = e.objects["obj2"].joints[0]
q_end = e.sim.data.get_joint_qpos(jname).copy()
sink = e.sink
Rs = T.euler2mat(np.array([0, 0, sink.rot]))
regs = regions_world(sink)
res["s19_regions"] = {k: dict(p0=r(g["p0"]), px=r(g["px"]), py=r(g["py"]), pz=r(g["pz"]), len=r(g["len"]),
                              tol_m=r([0.05 / L for L in g["len"]])) for k, g in regs.items()}
res["s19_sink"] = dict(pos=r(sink.pos), rot=float(sink.rot), size=r(sink.size), width=float(sink.width), depth=float(sink.depth))
res["s19_sponge_init_pos"] = r(sp_init_pos)
res["s19_sponge_end"] = contain_detail("obj2")
# sponge bbox half extents (body frame)
bb = e.objects["obj2"]._regions["bbox"]["elem"]
sp_half = string_to_array(bb.get("size")); sp_c = string_to_array(bb.get("pos"))
trials = []
for bname in [k for k in regs if k != "basin"] or ["basin"]:
    g = regs[bname]
    # region bounds in sink-local frame
    loc = lambda p: Rs.T @ (p - sink.pos)
    lo, hi = loc(g["p0"]), loc(g["p0"] + g["u"] + g["v"])
    xmin, xmax = min(lo[0], hi[0]), max(lo[0], hi[0]); ymin, ymax = min(lo[1], hi[1]), max(lo[1], hi[1])
    yc, xc = (ymin + ymax) / 2, (xmin + xmax) / 2
    for side in ("+x", "-x", "-y", "+y"):
        for dd in (0.03, 0.06, 0.10):
            # sponge centre placed so that its NEAR bbox face is dd outside the basin edge
            if side == "+x": pl = np.array([xmax + dd + sp_half[0], yc, 0])
            if side == "-x": pl = np.array([xmin - dd - sp_half[0], yc, 0])
            if side == "-y": pl = np.array([xc, ymin - dd - sp_half[1], 0])
            if side == "+y": pl = np.array([xc, ymax + dd + sp_half[1], 0])
            pw = sink.pos + Rs @ pl
            pw[2] = sp_init_pos[2]  # resting height on counter (initial sponge z)
            e.sim.set_state_from_flattened(end_state)
            q = q_end.copy(); q[:3] = pw - Rs @ sp_c * np.array([1, 1, 0]); q[3:] = sp_init_quat
            e.sim.data.set_joint_qpos(jname, q); e.sim.forward()
            inside = bool(OU.obj_inside_of(e, "obj2", sink, partial_check=False))
            det = contain_detail("obj2")
            regok = {k: v["inside"] for k, v in det["regions"].items()}
            # world-x/y distance between sponge bbox and basin edge in sink frame (actual, after placement)
            cl = [Rs.T @ (np.array(p) - sink.pos) for p in det["corners_world"]]
            trials.append(dict(basin=bname, side=side, gap_m=dd, sponge_pos=r(e.sim.data.body_xpos[sp_bid]),
                               corners_sinklocal_xy_min=r(np.min(cl, 0)[:2]), corners_sinklocal_xy_max=r(np.max(cl, 0)[:2]),
                               basin_local_x=[round(float(xmin), 4), round(float(xmax), 4)],
                               basin_local_y=[round(float(ymin), 4), round(float(ymax), 4)],
                               obj_inside_of=inside, per_region=regok,
                               step1_check=bool(STEPS[1]["check"](e, {})),
                               grip_far=bool(OU.gripper_obj_far(e, "obj2"))))
            print("s19 trial", trials[-1]["basin"], side, dd, inside, regok, flush=True)
res["s19_trials"] = trials
e.sim.set_state_from_flattened(end_state); e.sim.forward()
json.dump(res, open(out_path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wrote", out_path)
