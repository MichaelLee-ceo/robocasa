"""Bowl-fridge contact geoms and rack regions for StoreLeftovers s0,s14,s17 end states (read-only, CPU only)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state
RD = f"{ROOT}/runs/20261014_asus_seen11"; TASK = "StoreLeftoversInBowl"
env = robosuite.make(TASK, robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == TASK and "error" not in r}
out = []
for s in map(int, sys.argv[2:]):
    r = res[s]; z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    set_state(env, np.load(f"{RD}/{TASK}/{r['tag']}_end.npz", allow_pickle=True)["state"])
    m, d = env.sim.model, env.sim.data; fr = env.fridge
    bb = env.obj_body_id["bowl"]; bz = float(d.body_xpos[bb][2])
    geoms = set()
    for i in range(d.ncon):
        c = d.contact[i]
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if m.body_rootid[m.geom_bodyid[a]] == m.body_rootid[bb] and m.geom_bodyid[a] != m.geom_bodyid[b]:
                n = m.geom_id2name(b) or f"geom{b}"; bn = m.body_id2name(m.geom_bodyid[b])
                if "fridge" in (bn or ""): geoms.add(f"{n} @ {bn}")
    regs = {}
    for name in fr.get_reset_regions(env, compartment="fridge", reg_type=("shelf",)):
        reg = fr._regions[name]
        regs[name] = [round(100 * float(fr.pos[2] + reg["p0"][2]), 1), round(100 * float(fr.pos[2] + reg["pz"][2]), 1)]
    import robosuite.utils.transform_utils as T
    obj = env.objects["bowl"]; pos = np.array(d.body_xpos[bb]); quat = T.convert_quat(d.body_xquat[bb], to="xyzw")
    pts = obj.get_bbox_points(trans=pos, rot=quat); sites = fr.get_int_sites(relative=False); viol = {}
    for name in regs:
        p0, px, py, pz = sites[name]; v = []
        for ax, q in zip("xyz", (px, py, pz)):
            e = q - p0; L = np.linalg.norm(e); t = [float(np.dot(p - p0, e / L)) for p in pts]
            v.append(round(100 * max(0.0, -min(t), max(t) - L), 1))
        viol[name] = dict(zip(["out_x_cm", "out_y_cm", "out_z_cm"], v))
    row = dict(scene=f"s{s}", bbox_outside_by=viol, bowl_z_cm=round(100 * bz, 1), fridge_geoms=sorted(geoms), shelf_regions_z_cm=regs,
               rack_contact=bool(fr.check_rack_contact(env, "bowl")))
    out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
