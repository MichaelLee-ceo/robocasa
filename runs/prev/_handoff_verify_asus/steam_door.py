"""SteamInMicrowave end states: door normalized position and angle, is_closed/is_open, vegetable in bowl, bowl in
microwave (official / 0.05 m), gripper-to-start-button distance. Read-only, CPU only (robosuite.make, no renderer)."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T
from oracle import reset_to, set_state


def inside_fixed(env, obj_name, fixture, th=0.05):
    obj = env.objects[obj_name]
    pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
    quat = T.convert_quat(env.sim.data.body_xquat[env.obj_body_id[obj.name]], to="xyzw")
    pts = obj.get_bbox_points(trans=pos, rot=quat)
    for p0, px, py, pz in fixture.get_int_sites(relative=False).values():
        ok = True
        for q in (px, py, pz):
            e = q - p0; L = np.linalg.norm(e); eh = e / L
            if not all(-th <= np.dot(p - p0, eh) <= L + th for p in pts):
                ok = False; break
        if ok:
            return True
    return False


env = robosuite.make("SteamInMicrowave", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
rows = []
for rd in (f"{ROOT}/runs/20261014_asus_accept", f"{ROOT}/runs/20261014_asus_seen5"):
    recs = sorted((json.loads(l) for p in sorted(os.listdir(rd)) if p.startswith("results_w") for l in open(f"{rd}/{p}")), key=lambda r: r.get("seed", 0))
    for r in recs:
        if "error" in r or r["task"] != "SteamInMicrowave":
            continue
        z = np.load(r["scene"], allow_pickle=True)
        reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
        set_state(env, np.load(f"{rd}/SteamInMicrowave/{r['tag']}_end.npz", allow_pickle=True)["state"])
        mw = env.microwave; m, d = env.sim.model, env.sim.data
        js = mw.get_joint_state(env, mw.door_joint_names)
        jn = mw.door_joint_names[0]
        ang = float(np.degrees(d.qpos[m.get_joint_qpos_addr(jn)]))
        bid = m.geom_name2id(f"{mw.naming_prefix}start_button")
        eef = d.site_xpos[env.robots[0].eef_site_id["right"]]
        a = bool(OU.obj_inside_of(env, "bowl", mw)); b = inside_fixed(env, "bowl", mw)
        row = dict(scene=f"s{r['seed']}", ok=bool(r["success"]), door_norm=round(float(js[jn]), 4), door_deg=round(ang, 2),
                   is_closed=bool(mw.is_closed(env)), is_open=bool(mw.is_open(env)),
                   veg_in_bowl=bool(OU.check_obj_in_receptacle(env, "vegetable", "bowl")),
                   bowl_in_mw=("in" if a else "not in") + ("" if a == b else f" (official) / {'in' if b else 'not in'} (0.05 m)"),
                   grip_to_start_button_cm=round(100 * float(np.linalg.norm(eef - d.geom_xpos[bid])), 1),
                   done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(4)], end=r["steps"])
        rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
