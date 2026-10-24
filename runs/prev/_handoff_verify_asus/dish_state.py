"""LoadDishwasher end states for failed episodes (read-only, CPU only). Official rack check = contact between the dish's
own body geoms and the geoms of the rack1 joint's body (dishwasher.py:63-86); door closed = is_closed(th=0.05)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"; TASK = "LoadDishwasher"
env = robosuite.make(TASK, robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == TASK and "error" not in r}


def subtree(m, b):
    out = set()
    for i in range(m.nbody):
        j = i
        while j not in (0, b): j = m.body_parentid[j]
        if j == b: out.add(i)
    return out


def contacts(name):
    """bodies (not root names, so dishwasher parts are distinguishable) touching the object's subtree"""
    m, d = env.sim.model, env.sim.data; S = subtree(m, env.obj_body_id[name]); o = set()
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for a, b in ((b1, b2), (b2, b1)):
            if a in S and b not in S:
                n = m.body_id2name(b) or ""; r = m.body_id2name(m.body_rootid[b]) or ""
                o.add(n if "dishwasher" in n else r)
    return sorted(x for x in o if x)


rows = []
for s in sorted(res):
    r = res[s]
    if r["success"]:
        continue
    z = np.load(r["scene"], allow_pickle=True); meta = json.loads(str(z["ep_meta"]))
    reset_to(env, str(z["xml"]), meta, z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data; dw = env.dishwasher
    bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid])
    p0 = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in ("dish0", "dish1")}
    js = lambda: {k: round(float(v), 3) for k, v in dw.get_joint_state(env, list(dw._joint_names.values())).items()}
    j0 = js()
    set_state(env, np.load(f"{RD}/{TASK}/{r['tag']}_end.npz", allow_pickle=True)["state"])
    eef = d.site_xpos[env.robots[0].eef_site_id["right"]]; grip = env.robots[0].gripper["right"]
    rj = m.joint_name2id(dw._joint_names["rack"]); rack_pos = np.array(d.xpos[m.jnt_bodyid[rj]])
    objs = {}
    for o in ("dish0", "dish1"):
        p = np.array(d.body_xpos[env.obj_body_id[o]])
        objs[o] = dict(model=next(c["info"]["mjcf_path"].rstrip("/").split("/")[-2] for c in meta["object_cfgs"] if c["name"] == o),
                       rack_contact=bool(dw.check_rack_contact(env, o)), on=contacts(o),
                       z_cm=round(100 * float(p[2]), 1), to_rack_xy_cm=round(100 * float(np.linalg.norm(p[:2] - rack_pos[:2])), 1),
                       dz_rack_cm=round(100 * float(p[2] - rack_pos[2]), 1), moved_cm=round(100 * float(np.linalg.norm(p - p0[o])), 1),
                       grip_cm=round(100 * float(np.linalg.norm(eef - p)), 1),
                       grasped=bool(OU.check_obj_grasped(env, o)), touch_grip=bool(env.check_contact(grip, env.objects[o])))
    b1 = np.array(d.body_xpos[bid])
    row = dict(scene=f"s{s}", layout=meta.get("layout_id"), done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(3)],
               objs=objs, joints_start=j0, joints_end=js(), door_closed_005=bool(dw.is_closed(env, th=0.05)),
               base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1),
               base_to_dw_cm=round(100 * float(np.linalg.norm(b1[:2] - np.array(dw.pos)[:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
