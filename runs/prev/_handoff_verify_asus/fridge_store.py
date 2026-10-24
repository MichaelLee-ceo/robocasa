"""StoreLeftoversInBowl fridge geometry (all 30 scenes, read-only, CPU only, no env.step). Fridge frame = the fixture's
region frame (regions are stored relative to fridge.pos, rotated by fridge.rot); 'front' sign is taken from the robot
base's start position in that frame. Bowl bbox = obj.get_bbox_points at the end pose (same as obj_inside_of)."""
import json, os, re, sys
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"; TASK = "StoreLeftoversInBowl"
env = robosuite.make(TASK, robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == TASK and "error" not in r}
cm = lambda v: [round(100 * float(x), 1) for x in v]


def geom_pts(m, d, g):
    if m.geom_type[g] == 7:
        mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; v = m.mesh_vert[a:a + n]
    elif m.geom_type[g] == 6:  # box
        s = m.geom_size[g]; v = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
    else:
        return None
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


rows = []
for s in sorted(res):
    r = res[s]
    z = np.load(r["scene"], allow_pickle=True); meta = json.loads(str(z["ep_meta"])); xml = str(z["xml"])
    reset_to(env, xml, meta, z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data; fr = env.fridge
    Rf = T.euler2mat(np.array([0, 0, fr.rot])); fpos = np.array(fr.pos)
    loc = lambda w: Rf.T @ (np.asarray(w) - fpos)
    base = loc(d.body_xpos[m.body_name2id("mobilebase0_base")]); fs = 1.0 if base[1] > 0 else -1.0  # +1: front is +y
    obj = env.objects["bowl"]; bb = env.obj_body_id["bowl"]
    pts0 = obj.get_bbox_points(trans=np.array(d.body_xpos[bb]), rot=T.convert_quat(d.body_xquat[bb], to="xyzw"))
    bowl_d, bowl_h = 200 * float(obj.horizontal_radius), 100 * float(np.ptp(np.array(pts0)[:, 2]))
    eligible = set(fr.get_reset_regions(env, compartment="fridge"))
    regs = {}
    for name in fr.get_reset_region_names():
        if "fridge" not in name or "drawer" in name or name not in fr._regions:
            continue
        g = fr._regions[name]; p0, px, py, pz = (np.array(g[k]) for k in ("p0", "px", "py", "pz"))
        regs[name] = dict(x=cm([p0[0], px[0]]), y=cm(sorted([p0[1], py[1]])), z=cm([p0[2], pz[2]]),
                          abs_z=cm([fpos[2] + p0[2], fpos[2] + pz[2]]), eligible=name in eligible)
    order = sorted(regs, key=lambda n: regs[n]["z"][0])
    for i, n in enumerate(order):
        regs[n]["gap_to_next_region_cm"] = round(regs[order[i + 1]]["z"][0] - regs[n]["z"][1], 1) if i + 1 < len(order) else None
    glass = []
    for b in range(m.nbody):
        bn = m.body_id2name(b) or ""
        if bn.startswith(fr.naming_prefix) and "clear" in bn.lower():
            P = [geom_pts(m, d, g) for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            P = [loc(p) for q in P if q is not None for p in q]
            if P:
                P = np.array(P); glass.append(dict(body=bn, x=cm([P[:, 0].min(), P[:, 0].max()]), y=cm([P[:, 1].min(), P[:, 1].max()]),
                                                   z=cm([P[:, 2].min(), P[:, 2].max()])))
    set_state(env, np.load(f"{RD}/{TASK}/{r['tag']}_end.npz", allow_pickle=True)["state"])
    pts = np.array([loc(p) for p in obj.get_bbox_points(trans=np.array(d.body_xpos[bb]), rot=T.convert_quat(d.body_xquat[bb], to="xyzw"))])
    c = loc(d.body_xpos[bb]); bmin, bmax = pts.min(0), pts.max(0)
    inband = [n for n in order if regs[n]["z"][0] <= 100 * c[2] <= regs[n]["z"][0] + 0.9 * (regs[n]["z"][1] - regs[n]["z"][0])]
    marg = {}
    for n in order:
        g = regs[n]; y0, y1 = g["y"]; front, back = (y1, y0) if fs > 0 else (y0, y1)
        bf, bk = (100 * bmax[1], 100 * bmin[1]) if fs > 0 else (100 * bmin[1], 100 * bmax[1])
        marg[n] = dict(front=round(fs * (front - bf), 1), back=round(fs * (bk - back), 1),
                       left=round(100 * bmin[0] - g["x"][0], 1), right=round(g["x"][1] - 100 * bmax[0], 1),
                       bottom=round(100 * bmin[2] - g["z"][0], 1), top=round(g["z"][1] - 100 * bmax[2], 1))
    row = dict(scene=f"s{s}", ok=bool(r["success"]), layout=meta.get("layout_id"),
               fridge=";".join(sorted(set(re.findall(r'fixtures/fridges/([^/"]+)/', xml)))),
               front_is_plus_y=fs > 0, bowl_model=next(c_["info"]["mjcf_path"].rstrip("/").split("/")[-2] for c_ in meta["object_cfgs"] if c_["name"] == "bowl"),
               bowl_diam_cm=round(bowl_d, 1), bowl_h_cm=round(bowl_h, 1), regions=regs, glass=glass,
               bowl_end_centre=cm(c), bowl_end_bbox=dict(min=cm(bmin), max=cm(bmax)), centre_in_band=inband,
               margins_cm=marg, rack_contact=bool(fr.check_rack_contact(env, "bowl")))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
