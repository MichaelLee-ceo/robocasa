"""StirVegetables end states for failed episodes (read-only, CPU only)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state
from atomic_steps_seen11 import _picked, _released

RD = f"{ROOT}/runs/20261014_asus_seen11"
env = robosuite.make("StirVegetables", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "StirVegetables" and "error" not in r}


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
    pot0 = np.array(d.body_xpos[env.obj_body_id["pot"]])
    set_state(env, np.load(f"{RD}/StirVegetables/{r['tag']}_end.npz", allow_pickle=True)["state"])
    st = env.stove; knob = env.knob
    pp = np.array(d.body_xpos[env.obj_body_id["pot"]]); eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
    grip = env.robots[0].gripper["right"]
    veg = {}
    for o in ("veg1", "veg2"):
        p = np.array(d.body_xpos[env.obj_body_id[o]])
        inp = bool(OU.check_obj_in_receptacle(env, o, "pot"))
        if inp: loc = "in pot"
        elif OU.check_obj_grasped(env, o) or env.check_contact(grip, env.objects[o]): loc = "gripper"
        elif p[2] < 0.3: loc = "floor"
        else: loc = "on " + ",".join(contacts(o)) if contacts(o) else f"z {p[2]:.2f}"
        veg[o] = dict(loc=loc, xy_to_pot_cm=round(100 * float(np.linalg.norm(p[:2] - pp[:2])), 1),
                      th_cm=round(100 * env.objects["pot"].horizontal_radius * 0.7, 1), released=bool(_released(env, o)))
    sp = np.array(d.body_xpos[env.obj_body_id["spatula"]])
    burners = {k: np.array(d.get_site_xpos(v.get("name"))) for k, v in st.burner_sites.items() if v is not None}
    first = next((k for k in range(4) if dict((kk, t) for t, kk in r["history"]).get(k) is None), 3)
    target = {0: "veg1", 1: "veg2", 2: "spatula", 3: "veg1"}[first]
    b1 = np.array(d.body_xpos[bid])
    row = dict(scene=f"s{s}", done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(4)], named_knob=knob,
               veg=veg, pot_moved_cm=round(100 * float(np.linalg.norm(pp - pot0)), 1),
               pot_to_named_burner_cm=round(100 * float(np.linalg.norm(pp[:2] - burners[knob][:2])), 1) if knob in burners else None,
               named_knob_on=bool(st.is_burner_on(env=env, burner_loc=knob)),
               official_pot_loc=st.check_obj_location_on_stove(env, "pot", threshold=0.15),
               spatula_grasped=bool(OU.check_obj_grasped(env, "spatula")), step3_picked=bool(_picked(env, "spatula", env.counter)),
               spatula_on=contacts("spatula"), spatula_to_pot_cm=round(100 * float(np.linalg.norm(sp[:2] - pp[:2])), 1),
               spatula_touches=[o for o in ("veg1", "veg2") if env.check_contact(env.objects["spatula"], env.objects[o])],
               grip_to_target=target, grip_to_target_cm=round(100 * float(np.linalg.norm(eef - d.body_xpos[env.obj_body_id[target]])), 1),
               base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1),
               base_to_stove_cm=round(100 * float(np.linalg.norm(b1[:2] - np.array(st.pos)[:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
