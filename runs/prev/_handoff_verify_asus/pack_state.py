"""PackIdenticalLunches end states for failed episodes (read-only, CPU only)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"
FOODS = ["vegetable0", "vegetable1", "meat0", "meat1"]
env = robosuite.make("PackIdenticalLunches", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "PackIdenticalLunches" and "error" not in r}
rows = []
for s in sorted(res):
    r = res[s]
    if r["success"]:
        continue
    z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data
    fr = env.fridge
    try:
        dj = fr.door_joint_names; f0 = {k: round(float(v), 2) for k, v in fr.get_joint_state(env, dj).items()}
    except Exception:
        dj = None; f0 = None
    p0 = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in FOODS}
    bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid])
    set_state(env, np.load(f"{RD}/PackIdenticalLunches/{r['tag']}_end.npz", allow_pickle=True)["state"])
    grip = env.robots[0].gripper["right"]; eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
    tp = {t: np.array(d.body_xpos[env.obj_body_id[t]]) for t in ("tupperware0", "tupperware1")}
    foods = {}
    for o in FOODS:
        p = np.array(d.body_xpos[env.obj_body_id[o]])
        inn = [t for t in tp if OU.check_obj_in_receptacle(env, o, t)]
        if inn: loc = f"in {inn[0]}"
        elif OU.check_obj_grasped(env, o) or env.check_contact(grip, env.objects[o]): loc = "gripper"
        elif OU.obj_inside_of(env, o, fr): loc = "fridge"
        elif any(env.check_contact(env.objects[o], env.objects[t]) for t in tp): loc = "touching tupperware"
        elif p[2] < 0.3: loc = "floor"
        else: loc = f"elsewhere z {p[2]:.2f}"
        near = min(tp, key=lambda t: np.linalg.norm(p[:2] - tp[t][:2]))
        foods[o] = dict(loc=loc, moved_cm=round(100 * float(np.linalg.norm(p - p0[o])), 1),
                        nearest_tupperware=near, xy_to_it_cm=round(100 * float(np.linalg.norm(p[:2] - tp[near][:2])), 1),
                        th_cm=round(100 * env.objects[near].horizontal_radius * 0.7, 1),
                        grip_far=bool(OU.gripper_obj_far(env, o)))
    v0 = [o for o in ("vegetable0", "vegetable1") if OU.check_obj_in_receptacle(env, o, "tupperware0")]
    v1 = [o for o in ("vegetable0", "vegetable1") if OU.check_obj_in_receptacle(env, o, "tupperware1")]
    m0 = [o for o in ("meat0", "meat1") if OU.check_obj_in_receptacle(env, o, "tupperware0")]
    m1 = [o for o in ("meat0", "meat1") if OU.check_obj_in_receptacle(env, o, "tupperware1")]
    allo = v0 + v1 + m0 + m1
    nxt = next((o for o in FOODS if not foods[o]["loc"].startswith("in tupperware")), None)
    b1 = np.array(d.body_xpos[bid])
    row = dict(scene=f"s{s}", done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(4)],
               foods=foods, tupperware0=dict(veg=v0, meat=m0, valid=len(v0) == 1 and len(m0) == 1),
               tupperware1=dict(veg=v1, meat=m1, valid=len(v1) == 1 and len(m1) == 1),
               no_duplicates=len(allo) == len(set(allo)), grip_far_all_placed=all(OU.gripper_obj_far(env, o) for o in allo),
               fridge_doors_start=f0,
               fridge_doors_end=({k: round(float(v), 2) for k, v in fr.get_joint_state(env, dj).items()} if dj else None),
               grip_to_next_food_cm=round(100 * float(np.linalg.norm(eef - d.body_xpos[env.obj_body_id[nxt]])), 1) if nxt else None,
               next_food=nxt, base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1),
               base_to_fridge_cm=round(100 * float(np.linalg.norm(b1[:2] - np.array(fr.pos)[:2])), 1),
               base_to_tupperware0_cm=round(100 * float(np.linalg.norm(b1[:2] - tp["tupperware0"][:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
