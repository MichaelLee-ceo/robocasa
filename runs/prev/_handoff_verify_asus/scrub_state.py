"""ScrubCuttingBoard end states for failed episodes (read-only, CPU only). board_contact_positions / board_contact_timer
are Python-level state (not in the npz), so the number of scrub contacts and the sweep range are NOT recoverable here."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"; TASK = "ScrubCuttingBoard"
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


rows = []
for s in sorted(res):
    r = res[s]
    if r["success"]:
        continue
    z = np.load(r["scene"], allow_pickle=True); meta = json.loads(str(z["ep_meta"]))
    reset_to(env, str(z["xml"]), meta, z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data; bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid])
    P0 = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in ("sponge", "cutting_board")}
    st = dict(sponge_to_board_cm=round(100 * float(np.linalg.norm(P0["sponge"][:2] - P0["cutting_board"][:2])), 1),
              base_to_sponge_cm=round(100 * float(np.linalg.norm(b0[:2] - P0["sponge"][:2])), 1))
    set_state(env, np.load(f"{RD}/{TASK}/{r['tag']}_end.npz", allow_pickle=True)["state"])
    eef = d.site_xpos[env.robots[0].eef_site_id["right"]]; grip = env.robots[0].gripper["right"]
    sp, bp = np.array(d.body_xpos[env.obj_body_id["sponge"]]), np.array(d.body_xpos[env.obj_body_id["cutting_board"]])
    b1 = np.array(d.body_xpos[bid])
    sb = [m.geom_id2name(g) for g in range(m.ngeom) if m.geom_bodyid[g] == env.obj_body_id["sponge"]]
    bb = [m.geom_id2name(g) for g in range(m.ngeom) if m.geom_bodyid[g] == env.obj_body_id["cutting_board"]]
    row = dict(scene=f"s{s}", layout=meta.get("layout_id"), done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(2)],
               start=st, sponge_on=contacts("sponge"), sponge_board_contact=bool(env.check_contact(sb, bb)),
               sponge_grasped=bool(OU.check_obj_grasped(env, "sponge")), gripper_touch_sponge=bool(env.check_contact(grip, env.objects["sponge"])),
               grip_to_sponge_cm=round(100 * float(np.linalg.norm(eef - sp)), 1), gripper_far_015=bool(OU.gripper_obj_far(env, "sponge", th=0.15)),
               sponge_xy_to_board_cm=round(100 * float(np.linalg.norm(sp[:2] - bp[:2])), 1), sponge_dz_board_cm=round(100 * float(sp[2] - bp[2]), 1),
               sponge_moved_cm=round(100 * float(np.linalg.norm(sp - P0["sponge"])), 1), board_moved_cm=round(100 * float(np.linalg.norm(bp - P0["cutting_board"])), 1),
               base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
