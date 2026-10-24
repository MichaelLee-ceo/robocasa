"""GetToastedBread end states (read-only, CPU only): bread location, official end-state pieces, lever positions,
robot base distances. usage: python -I toast_state.py <out.json>"""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"
env = robosuite.make("GetToastedBread", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "GetToastedBread" and "error" not in r}


def slot_of(env):
    t = env.toaster
    for sp in t._slot_pairs:
        if t.check_slot_contact(env, "obj", slot_pair=sp):
            return sp
    return None


def levers(env):
    t = env.toaster; out = {}
    for sp in t._slot_pairs:
        jn = t._joint_names.get(f"lever_{sp}")
        if jn and jn in env.sim.model.joint_names:
            out[sp] = round(float(t.get_joint_state(env, [jn])[jn]), 3)
    return out


rows = []
for s in range(30):
    r = res[s]; z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    start_slot = slot_of(env)
    set_state(env, np.load(f"{RD}/GetToastedBread/{r['tag']}_end.npz", allow_pickle=True)["state"])
    d = env.sim.data
    bp = np.array(d.body_xpos[env.obj_body_id["obj"]]); pp = np.array(d.body_xpos[env.obj_body_id["plate"]])
    end_slot = slot_of(env)
    grip = env.robots[0].gripper["right"]
    held = bool(OU.check_obj_grasped(env, "obj")); touch = bool(env.check_contact(grip, env.objects["obj"]))
    on_plate = bool(OU.check_obj_in_receptacle(env, "obj", "plate"))
    if end_slot is not None: where = f"slot pair {end_slot}" + (" (start slot)" if end_slot == start_slot else f" (started in {start_slot})")
    elif held or touch: where = "in gripper" if held else "touching gripper"
    elif on_plate or env.check_contact(env.objects["obj"], env.objects["plate"]): where = "on plate"
    elif bp[2] < 0.3: where = "floor"
    else: where = "elsewhere (z %.2f m)" % bp[2]
    base = np.array(d.body_xpos[env.sim.model.body_name2id(env.robots[0].robot_model.root_body)])
    tp = np.array(env.toaster.pos)
    row = dict(scene=f"s{s}", ok=bool(r["success"]), done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(2)],
               bread=where, bread_plate_xy_cm=round(100 * float(np.linalg.norm(bp[:2] - pp[:2])), 1),
               on_plate_check=on_plate, gripper_far=bool(OU.gripper_obj_far(env, "obj")), levers=levers(env),
               base_to_plate_cm=round(100 * float(np.linalg.norm(base[:2] - pp[:2])), 1),
               base_to_toaster_cm=round(100 * float(np.linalg.norm(base[:2] - tp[:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
