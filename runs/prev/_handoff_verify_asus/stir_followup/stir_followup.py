"""StirVegetables follow-up (items 2-4). Read-only: reset_to/set_state + mj_forward only, no env.step. CPU only.
Methods:
  * geometry from collision geoms only (contype/conaffinity != 0); mesh vertices transformed to world with geom_xpos/geom_xmat
    (compiled mesh_vert is expressed in the geom frame); checked against geom_aabb.
  * min distances via mujoco.mj_geomDistance (distmax 1 m) over all collision-geom pairs; mesh-mesh uses convex hulls (MuJoCo convention).
  * pot inner bottom = max z of pot collision vertices within 0.5*R_int of pot body xy (R_int = pot_reg_int radius); rim = max z of all pot
    collision vertices within 1.3*R_int (excludes handles if they stick out further).  Also pot_reg_int cylinder bottom/top for reference.
  * spatula long axis = local body axis with largest extent of its collision vertices (body frame); tilt = angle between that axis and world z.
"""
import json, os, sys
import numpy as np, robosuite, mujoco
from scipy.spatial.transform import Rotation as Rot
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"
OUT = sys.argv[1]
SET3 = {8, 17, 22, 4, 10}
SET4 = {10, 11, 14, 15, 16, 23, 0, 1, 2, 3, 4, 5, 6, 12, 13, 21, 24, 29}
env = robosuite.make("StirVegetables", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}"))
       if r["task"] == "StirVegetables" and "error" not in r}
cm = lambda x: round(100 * float(x), 1)


def subtree(M, b):
    out = set()
    for i in range(M.nbody):
        j = i
        while j not in (0, b): j = M.body_parentid[j]
        if j == b: out.add(i)
    return out


def coll_geoms(M, name):
    S = subtree(M, env.obj_body_id[name])
    return [g for g in range(M.ngeom) if M.geom_bodyid[g] in S and (M.geom_contype[g] or M.geom_conaffinity[g])]


def verts_world(M, D, g):
    if M.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = M.geom_dataid[g]; a, n = M.mesh_vertadr[mid], M.mesh_vertnum[mid]
        v = M.mesh_vert[a:a + n]
    else:  # box corners fallback (aabb)
        c, h = M.geom_aabb[g][:3], M.geom_aabb[g][3:]
        v = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return D.geom_xpos[g] + v @ D.geom_xmat[g].reshape(3, 3).T


def aabb_check(M, g):
    if M.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: return 0.0
    mid = M.geom_dataid[g]; v = M.mesh_vert[M.mesh_vertadr[mid]:M.mesh_vertadr[mid] + M.mesh_vertnum[mid]]
    return float(np.abs((v.max(0) + v.min(0)) / 2 - M.geom_aabb[g][:3]).max() + np.abs((v.max(0) - v.min(0)) / 2 - M.geom_aabb[g][3:]).max())


def mind(M, D, G1, G2):
    best = (1e9, None); ft = np.zeros(6)
    for a in G1:
        for b in G2:
            d = mujoco.mj_geomDistance(M, D, a, b, 1.0, ft)
            if d < best[0]: best = (d, (a, b))
    return best


def contacts(name):
    m, d = env.sim.model, env.sim.data; S = subtree(m._model, env.obj_body_id[name]); o = set()
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in S and b2 not in S: o.add(m.body_id2name(m.body_rootid[b2]))
        elif b2 in S and b1 not in S: o.add(m.body_id2name(m.body_rootid[b1]))
    return sorted(x for x in o if x)


def spat_axis(M, D):
    b = env.obj_body_id["spatula"]; R = D.xmat[b].reshape(3, 3); p = D.xpos[b]
    V = np.vstack([verts_world(M, D, g) for g in coll_geoms(M, "spatula")])
    loc = (V - p) @ R  # body frame
    ext = loc.max(0) - loc.min(0); ax = int(np.argmax(ext))
    return ax, ext, np.array(R[:, ax])


rows = []
for s in sorted(res):
    r = res[s]
    z = np.load(r["scene"], allow_pickle=True); em = json.loads(str(z["ep_meta"]))
    reset_to(env, str(z["xml"]), em, z["state"]); env.unset_ep_meta()
    M, D = env.sim.model._model, env.sim.data._data; mujoco.mj_forward(M, D)
    PR0 = np.array(D.xmat[env.obj_body_id["pot"]]).reshape(3, 3)
    cfg = {c["name"]: c.get("info", {}) for c in em["object_cfgs"]}
    bid = env.sim.model.body_name2id("mobilebase0_base"); base = np.array(D.xpos[bid])
    P = {o: np.array(D.xpos[env.obj_body_id[o]]) for o in ("pot", "veg1", "veg2", "spatula")}
    q0 = np.array(D.xquat[env.obj_body_id["spatula"]]); ax0, ext0, axw0 = spat_axis(M, D)
    row = dict(seed=s, success=r["success"], steps=r["steps"], done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(4)],
               layout=em["layout_id"], style=em["style_id"], knob=env.knob, counter_ref=env.counter.name,
               models={o: cfg[o].get("mjcf_path", "").split("objects/")[-1].replace("/model.xml", "") for o in ("pot", "veg1", "veg2", "spatula")},
               cats={o: cfg[o].get("cat") for o in ("veg1", "veg2")},
               base_xy_cm={o: cm(np.linalg.norm(P[o][:2] - base[:2])) for o in P},
               spatula_on=contacts("spatula"), veg1_on=contacts("veg1"), veg2_on=contacts("veg2"),
               spatula_to_pot_xy_cm=cm(np.linalg.norm(P["spatula"][:2] - P["pot"][:2])),
               spatula_long_axis_local="xyz"[ax0], pot_tilt_start_deg=round(float(np.degrees(np.arccos(min(1.0, abs(PR0[2, 2]))))), 1), spatula_extent_cm=[cm(x) for x in ext0])
    row["veg_same_counter_as_spatula"] = bool(row["spatula_on"]) and row["veg1_on"] == row["spatula_on"] and row["veg2_on"] == row["spatula_on"]
    if s in SET3 or s in SET4:
        set_state(env, np.load(f"{RD}/StirVegetables/{r['tag']}_end.npz", allow_pickle=True)["state"]); mujoco.mj_forward(M, D)
        Sg = coll_geoms(M, "spatula"); E = {}
        sp1 = np.array(D.xpos[env.obj_body_id["spatula"]]); q1 = np.array(D.xquat[env.obj_body_id["spatula"]])
        ax1, _, axw1 = spat_axis(M, D)
        r0 = Rot.from_quat(q0[[1, 2, 3, 0]]); r1 = Rot.from_quat(q1[[1, 2, 3, 0]])
        yaw = lambda rr: rr.as_euler("ZYX", degrees=True)[0]
        hd = lambda a: np.degrees(np.arctan2(a[1], a[0]))
        dyaw = (yaw(r1) - yaw(r0) + 180) % 360 - 180; dhead = (hd(axw1) - hd(axw0) + 180) % 360 - 180
        E.update(spatula_dxy_cm=cm(np.linalg.norm(sp1[:2] - P["spatula"][:2])), spatula_dz_cm=cm(sp1[2] - P["spatula"][2]),
                 spatula_dyaw_deg_quat=round(float(dyaw), 1), spatula_dheading_longaxis_deg=round(float(dhead), 1),
                 spatula_rot_angle_total_deg=round(float(np.degrees((r1 * r0.inv()).magnitude())), 1))
        gr = env.robots[0].gripper["right"]
        jq = [float(D.qpos[M.jnt_qposadr[env.sim.model.joint_name2id(j)]]) for j in gr.joints]
        pads = [env.sim.model.geom_name2id(n) for n in ("gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision")]
        fing = [env.sim.model.geom_name2id(n) for n in ("gripper0_right_finger1_collision", "gripper0_right_finger2_collision")]
        dp, pp = mind(M, D, pads, Sg); df, _ = mind(M, D, pads + fing, Sg)
        E.update(gripper_qpos=[round(x, 4) for x in jq], gripper_opening_cm=cm(jq[0] - jq[1]),
                 pad_center_sep_cm=cm(np.linalg.norm(D.geom_xpos[pads[0]] - D.geom_xpos[pads[1]])),
                 min_pad_to_spatula_cm=round(100 * dp, 2), min_finger_or_pad_to_spatula_cm=round(100 * df, 2),
                 spatula_grasped_official=bool(OU.check_obj_grasped(env, "spatula")),
                 gripper_contacts_spatula=bool(env.check_contact(gr, env.objects["spatula"])),
                 eef_to_spatula_body_cm=cm(np.linalg.norm(D.site_xpos[env.robots[0].eef_site_id["right"]] - sp1)),
                 spatula_on_end=contacts("spatula"))
        # pot geometry, in the pot body frame (pots can be tilted at the end state)
        pbid = env.obj_body_id["pot"]; pb = np.array(D.xpos[pbid]); PR = np.array(D.xmat[pbid]).reshape(3, 3); Pg = coll_geoms(M, "pot")
        pot_tilt = float(np.degrees(np.arccos(min(1.0, abs(PR[2, 2])))))
        gi = env.sim.model.geom_name2id("pot_reg_int"); Rint = float(M.geom_size[gi][0]); hint = float(M.geom_size[gi][1])
        PL = (np.vstack([verts_world(M, D, g) for g in Pg]) - pb) @ PR; rr = np.linalg.norm(PL[:, :2], axis=1)
        inner_bottom = float(PL[rr < 0.5 * Rint, 2].max()); rim = float(PL[rr < 1.3 * Rint, 2].max())
        SW = np.vstack([verts_world(M, D, g) for g in Sg]); SL = (SW - pb) @ PR
        i = int(np.argmin(SL[:, 2])); low = float(SL[i, 2]); low_r = float(np.linalg.norm(SL[i, :2]))
        inside = np.linalg.norm(SL[:, :2], axis=1) < Rint
        # world-z variant: lowest spatula vertex vs world z of the inner-bottom point at pot axis and the highest rim vertex
        PW = np.vstack([verts_world(M, D, g) for g in Pg])
        wb = float((pb + PR @ np.array([0, 0, inner_bottom]))[2]); wr = float(PW[rr < 1.3 * Rint, 2].max())
        E.update(pot_tilt_deg=round(pot_tilt, 1), pot_inner_bottom_local_z_cm=cm(inner_bottom), pot_rim_local_z_cm=cm(rim), pot_depth_cm=cm(rim - inner_bottom),
                 pot_reg_int_local_bottom_top_cm=[cm(M.geom_pos[gi][2] - hint), cm(M.geom_pos[gi][2] + hint)], pot_R_int_cm=cm(Rint),
                 spatula_low_above_bottom_cm=cm(low - inner_bottom), spatula_low_below_rim_cm=cm(rim - low),
                 spatula_low_radial_cm=cm(low_r), spatula_frac_verts_inside_Rint=round(float(inside.mean()), 3),
                 spatula_low_inside_Rint_above_bottom_cm=cm(SL[inside, 2].min() - inner_bottom) if inside.any() else None,
                 world_spatula_low_z=round(float(SW[:, 2].min()), 4), world_pot_bottom_z=round(wb, 4), world_pot_rim_z=round(wr, 4),
                 world_spatula_low_above_bottom_cm=cm(SW[:, 2].min() - wb), world_spatula_low_below_rim_cm=cm(wr - SW[:, 2].min()))
        V1, V2 = coll_geoms(M, "veg1"), coll_geoms(M, "veg2")
        d1, _ = mind(M, D, Sg, V1); d2, _ = mind(M, D, Sg, V2); dpot, _ = mind(M, D, Sg, Pg)
        tilt = float(np.degrees(np.arccos(min(1.0, abs(axw1[2])))))
        E.update(min_spatula_veg1_cm=round(100 * d1, 2), min_spatula_veg2_cm=round(100 * d2, 2), min_spatula_pot_cm=round(100 * dpot, 2),
                 spatula_contacts_pot=bool(env.check_contact(env.objects["spatula"], env.objects["pot"])),
                 spatula_contacts_veg=[o for o in ("veg1", "veg2") if env.check_contact(env.objects["spatula"], env.objects[o])],
                 veg_in_pot=[o for o in ("veg1", "veg2") if OU.check_obj_in_receptacle(env, o, "pot")],
                 spatula_long_axis_local_end="xyz"[ax1], spatula_tilt_from_vertical_deg=round(tilt, 1),
                 mesh_aabb_maxerr=round(max(aabb_check(M, g) for g in Sg + Pg), 6))
        row["end"] = E
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
open(OUT + ".DONE", "w").write("ok\n")
