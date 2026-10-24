"""DeliverStraw end states for failed episodes (read-only, CPU only)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures.cabinets import Drawer
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"
env = robosuite.make("DeliverStraw", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "DeliverStraw" and "error" not in r}
rows = []
for s in sorted(res):
    r = res[s]
    if r["success"]:
        continue
    z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data
    drawers = {n: f for n, f in env.fixtures.items() if isinstance(f, Drawer)}
    st0 = {n: float(f.get_door_state(env)["door"]) for n, f in drawers.items()}
    cup0 = np.array(d.body_xpos[env.obj_body_id["glass_cup"]]); bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid])
    set_state(env, np.load(f"{RD}/DeliverStraw/{r['tag']}_end.npz", allow_pickle=True)["state"])
    st1 = {n: float(f.get_door_state(env)["door"]) for n, f in drawers.items()}
    dn = env.drawer.name
    sp = np.array(d.body_xpos[env.obj_body_id["straw"]]); cp = np.array(d.body_xpos[env.obj_body_id["glass_cup"]])
    grip = env.robots[0].gripper["right"]; eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
    touch_cup = bool(env.check_contact(env.objects["straw"], env.objects["glass_cup"]))
    if OU.check_obj_grasped(env, "straw") or env.check_contact(grip, env.objects["straw"]): loc = "gripper"
    elif touch_cup: loc = "touching cup"
    elif OU.obj_inside_of(env, "straw", env.drawer): loc = "in drawer"
    elif OU.check_obj_fixture_contact(env, "straw", env.dining_counter): loc = "on dining counter"
    elif sp[2] < 0.3: loc = "floor"
    else: loc = f"elsewhere z {sp[2]:.2f}"
    in_cup = bool(OU.check_obj_in_receptacle(env, "straw", "glass_cup", th=0.5))
    gfar = bool(OU.gripper_obj_far(env, "straw"))
    step1 = all(v >= 0.95 for v in env.drawer.get_door_state(env=env).values())
    first_open = dict((k, t) for t, k in r["history"]).get(0)
    target = env.drawer.pos if first_open is None else sp
    b1 = np.array(d.body_xpos[bid])
    row = dict(scene=f"s{s}", done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(2)],
               drawer_open_start=round(st0[dn], 3), drawer_open_end=round(st1[dn], 3), step1_drawer_ge_0p95=step1,
               other_drawers_open_gt5pct={n: round(v, 3) for n, v in st1.items() if n != dn and v > 0.05},
               other_drawers_changed={n: [round(st0[n], 3), round(v, 3)] for n, v in st1.items() if n != dn and abs(v - st0[n]) > 0.02},
               straw=loc, straw_cup_xy_cm=round(100 * float(np.linalg.norm(sp[:2] - cp[:2])), 1), straw_cup_contact=touch_cup,
               official_in_cup_th0p5=in_cup, official_gripper_far_0p25=gfar,
               cup_moved_cm=round(100 * float(np.linalg.norm(cp - cup0)), 1), cup_upright=bool(OU.check_obj_upright(env, "glass_cup")),
               grip_to_next_target_cm=round(100 * float(np.linalg.norm(eef - np.array(target))), 1),
               grip_target="drawer" if first_open is None else "straw",
               base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1),
               base_to_drawer_cm=round(100 * float(np.linalg.norm(b1[:2] - np.array(env.drawer.pos)[:2])), 1),
               base_to_cup_cm=round(100 * float(np.linalg.norm(b1[:2] - cp[:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
