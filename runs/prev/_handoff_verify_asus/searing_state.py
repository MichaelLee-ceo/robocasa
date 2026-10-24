"""SearingMeat end states (read-only, CPU only): named burner, stove group, pan/meat location and check pieces,
knob values, gripper-to-knob distance, base movement. usage: python -I searing_state.py <out.json>"""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state
from atomic_steps_seen11 import _pan_on_knob_burner, _released

RD = f"{ROOT}/runs/20261014_asus_seen11"
SX = json.load(open(f"{ROOT}/runs/_handoff/verify_asus/stove_x.json"))
env = robosuite.make("SearingMeat", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "SearingMeat" and "error" not in r}
rows = []
for s in range(30):
    r = res[s]; z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data
    st = env.stove; knob = env.knob
    sb = m.body_name2id(st.root_body); Rm = d.body_xmat[sb].reshape(3, 3); c = d.body_xpos[sb]
    X = lambda p: float((Rm.T @ (np.array(p) - c))[0])
    kanch = {k: np.array(d.xanchor[m.joint_name2id(f"{st.naming_prefix}knob_{k}_joint")]) for k, j in st.knob_joints.items() if j is not None}
    kx = {k: round(100 * X(v), 1) for k, v in kanch.items()}
    order = sorted(kx, key=kx.get)
    grp = next((g for g, v in SX.items() if sorted(v["knobs_x"]) == sorted(kx) and all(abs(v["knobs_x"][k] - kx[k]) < 0.6 for k in kx)), "new")
    bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid])
    meat0 = np.array(d.body_xpos[env.obj_body_id["meat"]])
    set_state(env, np.load(f"{RD}/SearingMeat/{r['tag']}_end.npz", allow_pickle=True)["state"])
    pp = np.array(d.body_xpos[env.obj_body_id["pan"]]); mp = np.array(d.body_xpos[env.obj_body_id["meat"]])
    burners = {k: np.array(d.get_site_xpos(v.get("name"))) for k, v in st.burner_sites.items() if v is not None}
    grip = env.robots[0].gripper["right"]
    # pan location
    if OU.check_obj_fixture_contact(env, "pan", st):
        nb = min(burners, key=lambda k: np.linalg.norm(burners[k][:2] - pp[:2]))
        pan_loc = f"stove (nearest {nb})"
    elif OU.check_obj_grasped(env, "pan") or env.check_contact(grip, env.objects["pan"]): pan_loc = "gripper"
    elif OU.obj_inside_of(env, "pan", env.cab): pan_loc = "cabinet"
    elif pp[2] < 0.3: pan_loc = "floor"
    else: pan_loc = f"elsewhere z {pp[2]:.2f}"
    pan_named_cm = round(100 * float(np.linalg.norm(burners[knob][:2] - pp[:2])), 1) if knob in burners else None
    # meat location
    in_pan = bool(OU.check_obj_in_receptacle(env, "meat", "pan", th=0.07))
    if in_pan: meat_loc = "in pan"
    elif OU.check_obj_grasped(env, "meat") or env.check_contact(grip, env.objects["meat"]): meat_loc = "gripper"
    elif np.linalg.norm(mp - meat0) < 0.03: meat_loc = "at start"
    elif env.check_contact(env.objects["meat"], env.objects["pan"]): meat_loc = "touching pan"
    elif mp[2] < 0.3: meat_loc = "floor"
    else: meat_loc = f"elsewhere z {mp[2]:.2f}"
    ks = st.get_knobs_state(env=env)
    on = [k for k in order if st.is_burner_on(env=env, burner_loc=k)]
    eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
    b1 = np.array(d.body_xpos[bid])
    row = dict(scene=f"s{s}", ok=bool(r["success"]), done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(3)],
               layout=json.loads(str(z["ep_meta"])).get("layout_id"), named=knob, group=grp,
               pan=pan_loc, pan_to_named_cm=pan_named_cm, step1_pan_on_named=bool(_pan_on_knob_burner(env)),
               step1_grip_far=bool(OU.gripper_obj_far(env, "pan")),
               meat=meat_loc, meat_to_pan_cm=round(100 * float(np.linalg.norm(mp[:2] - pp[:2])), 1), step2_in_pan=in_pan,
               step2_released=bool(_released(env, "meat")),
               knobs={k: round(float(ks[k]), 2) for k in order}, on=on, named_on=knob in on,
               on_rank_from_left=[order.index(k) + 1 for k in on],
               grip_to_named_knob_cm=round(100 * float(np.linalg.norm(eef - kanch[knob])), 1) if knob in kanch else None,
               base_moved_cm=round(100 * float(np.linalg.norm(b1[:2] - b0[:2])), 1),
               base_to_stove_cm=round(100 * float(np.linalg.norm(b1[:2] - np.array(st.pos)[:2])), 1),
               official_pan_loc=st.check_obj_location_on_stove(env, "pan", threshold=0.15),
               meat_grip_far=bool(OU.gripper_obj_far(env, "meat")))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
