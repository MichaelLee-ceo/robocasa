"""(A) s23/s27 pan containment detail at the end state (like s13); (B) per-scene table s0-29: layout/style, sink
fixture name + model, handle joint, ASUS + hscc results, ASUS end-state water_on / containment.
usage: python -I audit_extra.py <out.json> <out.csv>"""
import csv, glob, json, os, re, sys
import numpy as np
import gymnasium as gym

ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import groot_eval  # noqa
from oracle import reset_to, set_state
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T

RUN = f"{ROOT}/runs/20261014_asus_seen5"
HSCC = f"{ROOT}/runs/20261013_scenes_v2"
out_json, out_csv = sys.argv[1], sys.argv[2]
e = gym.make("robocasa/PreSoakPan", split="target", enable_render=False).unwrapped.env


def recs(d, cond):
    R = {}
    for p in sorted(glob.glob(f"{d}/results_w*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r and r.get("task") == "PreSoakPan" and r.get("condition") == cond:
                R[int(r["seed"])] = r
    return R


A, H = recs(RUN, "baseline_seed"), recs(HSCC, "baseline_seed_save")


def r4(x):
    return np.round(np.asarray(x, dtype=float), 4).tolist()


def contain_detail(name, th=0.05):
    obj = e.objects[name]
    pos = np.array(e.sim.data.body_xpos[e.obj_body_id[obj.name]])
    quat = T.convert_quat(e.sim.data.body_xquat[e.obj_body_id[obj.name]], to="xyzw")
    pts = obj.get_bbox_points(trans=pos, rot=quat); Rm = T.quat2mat(quat)
    out = dict(obj_pos=r4(pos), obj_quat_xyzw=r4(quat), tilt_deg=round(float(np.degrees(np.arccos(np.clip(Rm[2, 2], -1, 1)))), 1),
               corners_objlocal=[r4(Rm.T @ (p - pos)) for p in pts], corners_world=[r4(p) for p in pts],
               lib=bool(OU.obj_inside_of(e, name, e.sink, partial_check=False)),
               center=bool(OU.obj_inside_of(e, name, e.sink, partial_check=True)), regions={})
    for rn, (p0, px, py, pz) in e.sink.get_int_sites(relative=False).items():
        bad = []
        for i, p in enumerate(pts):
            for ax, hi in (("u", px), ("v", py), ("w", pz)):
                a = hi - p0; L = np.linalg.norm(a); ah = a / L; tol = th / L
                s, lo_, hh = ah @ p, ah @ p0, ah @ hi
                m_lo, m_hi = s - (lo_ - tol), (hh + tol) - s
                if m_lo < 0 or m_hi < 0:
                    bad.append(dict(corner=i, axis=ax, region_len=round(float(L), 4), tol_m=round(float(tol), 4),
                                    outside_by_m=round(float(-min(m_lo, m_hi)), 4), side="lo" if m_lo < 0 else "hi"))
        out["regions"][rn] = dict(len=r4([np.linalg.norm(px - p0), np.linalg.norm(py - p0), np.linalg.norm(pz - p0)]),
                                  failing=bad, inside=not bad)
    return out


def mesh_profile(name, nb=10):
    m, d = e.sim.model, e.sim.data
    bid = e.obj_body_id[name]; pos, R = d.body_xpos[bid], d.body_xmat[bid].reshape(3, 3)
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
            mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V.append(((m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - pos) @ R)
    V = np.concatenate(V); ext = V.max(0) - V.min(0); la = int(np.argmax(ext[:2])); sa = 1 - la
    edges = np.linspace(V[:, la].min(), V[:, la].max(), nb + 1); prof = []
    for i in range(nb):
        sel = (V[:, la] >= edges[i]) & (V[:, la] <= edges[i + 1])
        prof.append(dict(bin=[round(float(edges[i]), 3), round(float(edges[i + 1]), 3)],
                         width=round(float(np.ptp(V[sel, sa])) if sel.sum() > 1 else 0.0, 3),
                         center=round(float(V[sel, sa].mean()) if sel.sum() else 0.0, 3)))
    return dict(long_axis="xyz"[la], vert_min=r4(V.min(0)), vert_max=r4(V.max(0)), profile=prof)


res = dict(detail={}, table=[])
rows = []
for s in range(30):
    r = A.get(s)
    z = np.load(r["scene"], allow_pickle=True)
    meta = json.loads(str(z["ep_meta"])); xml = str(z["xml"])
    sink_name = meta["fixture_refs"]["sink"]
    files = sorted(set(re.findall(r'file="[^"]*/fixtures/sinks/([^/"]+)/', xml)))
    hj = re.findall(rf'<joint[^>]*name="{sink_name}_handle_joint"[^>]*>', xml)
    rng = re.search(r'range="([^"]*)"', hj[0]).group(1) if hj else None
    axis = re.search(r'axis="([^"]*)"', hj[0]).group(1) if hj else None
    reset_to(e, xml, meta, z["state"]); e.unset_ep_meta()
    set_state(e, np.load(f"{RUN}/PreSoakPan/PreSoakPan_s{s}_single_end.npz", allow_pickle=True)["state"])
    hs = e.sink.get_handle_state(env=e)
    dbl = "basin_left" in e.sink._regions
    row = dict(seed=s, layout=meta["layout_id"], style=meta["style_id"], sink_fixture=sink_name,
               sink_class=type(e.sink).__name__, sink_model=";".join(files), double_basin=dbl,
               handle_joint=f"{sink_name}_handle_joint", handle_range=rng, handle_axis=axis,
               pan=r["scene_fp"]["objects"]["obj1"].split("/")[2], sponge=r["scene_fp"]["objects"]["obj2"].split("/")[2],
               asus_ok=r["success"], asus_reached=r["reached"], asus_end=r["steps"],
               asus_history=" ".join(f"{k}@{t}" for t, k in r["history"]),
               asus_end_handle_qpos=round(float(hs["handle_joint"]), 4), asus_end_water_on=bool(hs["water_on"]),
               asus_end_pan_in_sink=bool(OU.obj_inside_of(e, "obj1", e.sink, partial_check=False)),
               asus_end_sponge_in_sink=bool(OU.obj_inside_of(e, "obj2", e.sink, partial_check=False)))
    h = H.get(s)
    row.update(hscc_ok=(h["success"] if h else ""), hscc_reached=(h["reached"] if h else ""), hscc_end=(h["steps"] if h else ""),
               hscc_history=(" ".join(f"{k}@{t}" for t, k in h["history"]) if h else ""))
    rows.append(row); print(json.dumps(row), flush=True)
    if s in (23, 27):
        d = contain_detail("obj1"); d["mesh_profile"] = mesh_profile("obj1")
        d["basin_loc_full"] = e.sink.get_obj_basin_loc(e, "obj1", partial_check=False) if dbl else None
        d["basin_loc_center"] = e.sink.get_obj_basin_loc(e, "obj1", partial_check=True) if dbl else None
        d["sponge"] = contain_detail("obj2")
        res["detail"][s] = d
        print("detail", s, json.dumps({k: v for k, v in d.items() if k not in ("sponge",)}), flush=True)
res["table"] = rows
json.dump(res, open(out_json, "w"), indent=1)
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", out_json, out_csv)
