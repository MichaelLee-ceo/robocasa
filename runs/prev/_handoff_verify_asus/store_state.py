"""StoreLeftoversInBowl end states for failed episodes (read-only, CPU only)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"
TASK = "StoreLeftoversInBowl"
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
    m, d = env.sim.model, env.sim.data; S = subtree(m, env.obj_body_id[name]); o = set()
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in S and b2 not in S: o.add(m.body_id2name(m.body_rootid[b2]))
        elif b2 in S and b1 not in S: o.add(m.body_id2name(m.body_rootid[b1]))
    return sorted(x for x in o if x)


def safe(f):
    try: return f()
    except Exception as e: return f"err {e!r}"[:60]


rows = []
for s in sorted(res):
    r = res[s]
    if r["success"]:
        continue
    z = np.load(r["scene"], allow_pickle=True)
    meta = json.loads(str(z["ep_meta"]))
    reset_to(env, str(z["xml"]), meta, z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data; bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid])
    names = ("chicken_drumstick", "vegetable", "bowl")
    p0 = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in names}
    door0 = safe(lambda: env.fridge.is_open(env))
    set_state(env, np.load(f"{RD}/{TASK}/{r['tag']}_end.npz", allow_pickle=True)["state"])
    grip = env.robots[0].gripper["right"]; eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
    bp = np.array(d.body_xpos[env.obj_body_id["bowl"]])
    objs = {}
    for o in names:
        p = np.array(d.body_xpos[env.obj_body_id[o]])
        e = dict(on=contacts(o), moved_cm=round(100 * float(np.linalg.norm(p - p0[o])), 1),
                 grip_cm=round(100 * float(np.linalg.norm(eef - p)), 1),
                 grasped=bool(OU.check_obj_grasped(env, o)), touch_grip=bool(env.check_contact(grip, env.objects[o])))
        if o != "bowl":
            e["in_bowl"] = bool(OU.check_obj_in_receptacle(env, o, "bowl"))
            e["xy_to_bowl_cm"] = round(100 * float(np.linalg.norm(p[:2] - bp[:2])), 1)
            e["dz_to_bowl_cm"] = round(100 * float(p[2] - bp[2]), 1)
        else:
            e["rack_contact"] = bool(env.fridge.check_rack_contact(env, "bowl"))
            e["inside_fridge"] = safe(lambda: bool(OU.obj_inside_of(env, "bowl", env.fridge)))
            e["xy_to_fridge_cm"] = round(100 * float(np.linalg.norm(p[:2] - np.array(env.fridge.pos)[:2])), 1)
            e["z_cm"] = round(100 * float(p[2]), 1)
            e["gripper_far"] = bool(OU.gripper_obj_far(env, "bowl"))
        objs[o] = e
    b1 = np.array(d.body_xpos[bid])
    row = dict(scene=f"s{s}", layout=meta.get("layout_id"), veg=env.objects["vegetable"].name,
               done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(3)], objs=objs,
               fridge_open_start_end=[door0, safe(lambda: env.fridge.is_open(env))],
               base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1),
               base_to_fridge_cm=round(100 * float(np.linalg.norm(b1[:2] - np.array(env.fridge.pos)[:2])), 1),
               base_to_bowl_cm=round(100 * float(np.linalg.norm(b1[:2] - bp[:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
