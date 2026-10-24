"""SetUpCuttingStation start geometry for all 30 scenes plus end distances (read-only, CPU only, no env.step).
Knife long axis = principal xy axis of the knife's mesh vertices (world frame); blade end = the end whose vertices span
less height (blade is flat, handle is thicker). Drawer frame = drawer root body frame; 'front' = the side the robot is on."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"; TASK = "SetUpCuttingStation"
env = robosuite.make(TASK, robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == TASK and "error" not in r}


def subtree(m, b):
    out = set()
    for i in range(m.nbody):
        j = i
        while j not in (0, b): j = m.body_parentid[j]
        if j == b: out.add(i)
    return out


def verts(name):
    m, d = env.sim.model, env.sim.data; S = subtree(m, env.obj_body_id[name]); V = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in S and m.geom_type[g] == 7:  # mesh
            mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n]; R = d.geom_xmat[g].reshape(3, 3)
            V.append(v @ R.T + d.geom_xpos[g])
    return np.concatenate(V) if V else None


def wrap(a):
    return float((a + 180) % 360 - 180)


def side(v):  # base-frame xy direction -> words (base +x = forward, +y = left)
    a = np.degrees(np.arctan2(v[1], v[0]))
    return ["forward", "forward-left", "left", "back-left", "back", "back-right", "right", "forward-right"][int(((a + 22.5) % 360) // 45)]


rows = []
for s in sorted(res):
    r = res[s]
    z = np.load(r["scene"], allow_pickle=True); meta = json.loads(str(z["ep_meta"]))
    reset_to(env, str(z["xml"]), meta, z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data
    model = {c["name"]: (c["info"]["cat"], c["info"]["mjcf_path"].rstrip("/").split("/")[-2]) for c in meta["object_cfgs"]}
    bid = m.body_name2id("mobilebase0_base"); b0 = np.array(d.body_xpos[bid]); Rb = d.body_xmat[bid].reshape(3, 3)
    P = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in ("knife", "meat", "receptacle")}
    bear = lambda p: round(wrap(np.degrees(np.arctan2(*(Rb.T @ (p - b0))[[1, 0]]))), 0)
    xy = lambda a, b: round(100 * float(np.linalg.norm(a[:2] - b[:2])), 1)
    # knife axis
    V = verts("knife"); c = V[:, :2].mean(0); U = V[:, :2] - c
    ax = np.linalg.svd(U, full_matrices=False)[2][0]; t = U @ ax; L = t.max() - t.min()
    lo, hi = t < t.min() + 0.25 * L, t > t.max() - 0.25 * L
    hz = lambda mk: float(V[mk, 2].max() - V[mk, 2].min())
    blade = ax if hz(hi) < hz(lo) else -ax  # points from handle to blade tip
    bl = Rb.T @ np.array([blade[0], blade[1], 0.0])
    yaw = wrap(np.degrees(np.arctan2(bl[1], bl[0])))
    # drawer frame offset of the knife from the drawer interior centre
    dr = env.drawer; did = m.body_name2id(dr.root_body); Rd = d.body_xmat[did].reshape(3, 3); dp = d.body_xpos[did]
    regs = dr.get_int_sites(relative=False); p0, px, py, pz = list(regs.values())[0]
    cen = p0 + 0.5 * (px - p0) + 0.5 * (py - p0)
    k_loc = Rd.T @ (P["knife"] - cen); r_loc = Rd.T @ (b0 - dp)
    front_sign = 1.0 if r_loc[1] > 0 else -1.0
    width, depth = float(np.linalg.norm(px - p0)), float(np.linalg.norm(py - p0))
    row = dict(scene=f"s{s}", ok=bool(r["success"]), layout=meta.get("layout_id"),
               knife=model["knife"][1], meat=f"{model['meat'][0]} ({model['meat'][1]})", board=model["receptacle"][1],
               knife_len_cm=round(100 * L, 1),
               knife_board=xy(P["knife"], P["receptacle"]), meat_board=xy(P["meat"], P["receptacle"]),
               base_knife=xy(b0, P["knife"]), base_board=xy(b0, P["receptacle"]), base_meat=xy(b0, P["meat"]),
               bearing_board_deg=bear(P["receptacle"]), bearing_meat_deg=bear(P["meat"]), bearing_knife_deg=bear(P["knife"]),
               knife_dz_base_cm=round(100 * float(P["knife"][2] - b0[2]), 1),
               blade_yaw_deg=round(yaw, 0), blade_points=side(bl),
               drawer_offset_right_cm=round(100 * float(k_loc[0]), 1), drawer_offset_front_cm=round(100 * float(front_sign * k_loc[1]), 1),
               drawer_int_w_d_cm=[round(100 * width, 1), round(100 * depth, 1)],
               drawer_open_start=round(float(dr.get_door_state(env)["door"]), 2))
    set_state(env, np.load(f"{RD}/{TASK}/{r['tag']}_end.npz", allow_pickle=True)["state"])
    b1 = np.array(d.body_xpos[bid]); E = {o: np.array(d.body_xpos[env.obj_body_id[o]]) for o in ("knife", "meat", "receptacle")}
    row["base_moved_cm"] = xy(b1, b0)
    if r["success"]:
        row["end_knife_board"] = xy(E["knife"], E["receptacle"]); row["end_meat_board"] = xy(E["meat"], E["receptacle"])
        row["th_cm"] = round(100 * env.objects["receptacle"].horizontal_radius * 0.7, 1)
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
