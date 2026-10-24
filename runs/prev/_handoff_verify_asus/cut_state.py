"""SetUpCuttingStation end states for failed episodes (read-only, CPU only)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures.cabinets import Drawer
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"
env = robosuite.make("SetUpCuttingStation", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "SetUpCuttingStation" and "error" not in r}


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


rows = []
for s in sorted(res):
    r = res[s]
    if r["success"]:
        continue
    z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data; bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid])
    drawers = {n: f for n, f in env.fixtures.items() if isinstance(f, Drawer)}
    st0 = {n: float(f.get_door_state(env)["door"]) for n, f in drawers.items()}
    p0 = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in ("knife", "meat", "receptacle")}
    set_state(env, np.load(f"{RD}/SetUpCuttingStation/{r['tag']}_end.npz", allow_pickle=True)["state"])
    st1 = {n: float(f.get_door_state(env)["door"]) for n, f in drawers.items()}
    grip = env.robots[0].gripper["right"]; eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
    rp = np.array(d.body_xpos[env.obj_body_id["receptacle"]])
    th = env.objects["receptacle"].horizontal_radius * 0.7
    objs = {}
    for o in ("knife", "meat"):
        p = np.array(d.body_xpos[env.obj_body_id[o]])
        onb = bool(OU.check_obj_in_receptacle(env, o, "receptacle"))
        if onb: loc = "on board"
        elif OU.check_obj_grasped(env, o) or env.check_contact(grip, env.objects[o]): loc = "gripper"
        elif o == "knife" and OU.obj_inside_of(env, o, env.drawer): loc = "in drawer"
        elif p[2] < 0.3: loc = "floor"
        else: loc = "on " + ",".join(contacts(o))
        objs[o] = dict(loc=loc, xy_to_board_cm=round(100 * float(np.linalg.norm(p[:2] - rp[:2])), 1), th_cm=round(100 * th, 1),
                       contact_board=bool(env.check_contact(env.objects[o], env.objects["receptacle"])),
                       moved_cm=round(100 * float(np.linalg.norm(p - p0[o])), 1), released=not env.check_contact(grip, env.objects[o]))
    dn = env.drawer.name
    first = next((k for k in range(2) if dict((kk, t) for t, kk in r["history"]).get(k) is None), 1)
    tgt = "knife" if first == 0 else "meat"
    b1 = np.array(d.body_xpos[bid])
    row = dict(scene=f"s{s}", done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(2)], objs=objs,
               board_moved_cm=round(100 * float(np.linalg.norm(rp - p0["receptacle"])), 1),
               grip_far_knife=bool(OU.gripper_obj_far(env, "knife")), grip_far_board=bool(OU.gripper_obj_far(env, "receptacle")),
               drawer_open_start_end=[round(st0[dn], 2), round(st1[dn], 2)],
               other_drawers_changed={n: [round(st0[n], 2), round(v, 2)] for n, v in st1.items() if n != dn and abs(v - st0[n]) > 0.05},
               grip_to=tgt, grip_to_cm=round(100 * float(np.linalg.norm(eef - d.body_xpos[env.obj_body_id[tgt]])), 1),
               base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1),
               base_to_drawer_cm=round(100 * float(np.linalg.norm(b1[:2] - np.array(env.drawer.pos)[:2])), 1),
               base_to_board_cm=round(100 * float(np.linalg.norm(b1[:2] - rp[:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
