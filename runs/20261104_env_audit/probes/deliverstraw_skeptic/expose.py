"""For frozen DeliverStraw scenes: open the drawer to several official fractions (straw riding along with the drawer),
and measure how much of the straw is visible from straight above (vertical ray hits the straw first, robot ignored).
Also reports the fraction at which the straw's full length clears the cabinet top."""
import os, sys, json
import numpy as np
import gymnasium as gym
import mujoco
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import atomic_eval  # noqa  registers envs
from atomic_eval import load_scene
import robocasa.utils.object_utils as OU

scenes = sys.argv[1:] or ["s0", "s17", "s25", "s27", "s7"]
env = gym.make("robocasa/DeliverStraw", split="target", enable_render=False).unwrapped.env
m, d = env.sim.model._model, env.sim.data._data
for sc in scenes:
    s0 = load_scene(env, f"{R}/runs/scenes_seen11_v2/DeliverStraw_{sc}.npz")
    m, d = env.sim.model._model, env.sim.data._data
    dr = env.drawer
    jname = f"{dr.name}_slidejoint"
    jadr = env.sim.model.get_joint_qpos_addr(jname)
    jmax = dr.size[1] * 0.55
    straw_bid = env.obj_body_id["straw"]
    sj = env.objects["straw"].joints[0]
    sadr = env.sim.model.get_joint_qpos_addr(sj)[0]
    dbody = m.jnt_bodyid[env.sim.model.joint_name2id(jname)]
    # pull direction: drawer body displacement per unit of -qpos
    env.sim.set_state_from_flattened(s0); env.sim.forward()
    p0 = d.xpos[dbody].copy()
    env.sim.data.qpos[jadr] = -0.1; env.sim.forward()
    pull = (d.xpos[dbody] - p0) / 0.1
    straw_geoms = [env.sim.model.geom_name2id(g) for g in env.objects["straw"].contact_geoms]
    robot_bodies = {i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith(("robot0_", "mobilebase0_", "gripper0_"))}
    res = dict(scene=sc, joint_range=[float(x) for x in m.jnt_range[env.sim.model.joint_name2id(jname)]], jmax=float(jmax),
               pull=pull.round(3).tolist(), rows=[])
    for f in [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.2, 1.5, 1.82]:
        env.sim.set_state_from_flattened(s0); env.sim.forward()
        q = env.sim.data.qpos
        delta = f * jmax
        q[jadr] = -delta
        q[sadr:sadr + 3] = q[sadr:sadr + 3] + pull * delta
        env.sim.forward()
        val = dr.get_door_state(env)["door"]
        # sample points on the straw: vertices of its collision geoms' aabbs in world frame
        pts = []
        for g in straw_geoms:
            c, hs = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            xm = d.geom_xmat[g].reshape(3, 3); xp = d.geom_xpos[g]
            for sx in np.linspace(-1, 1, 5):
                for sy in np.linspace(-1, 1, 5):
                    pts.append(xp + xm @ (c + hs * np.array([sx, sy, 1.0])))
        pts = np.array(pts)
        vis = 0
        for p in pts:
            start = np.array([p[0], p[1], p[2] + 1.5])
            gid = np.array([-1], dtype=np.int32)
            hit_straw = False
            for _ in range(20):
                dist = mujoco.mj_ray(m, d, start, np.array([0, 0, -1.0]), None, 1, -1, gid)
                if dist < 0 or gid[0] < 0:
                    break
                b = m.geom_bodyid[gid[0]]
                if gid[0] in straw_geoms:
                    hit_straw = True; break
                rb = m.body_rootid[b]
                if b in robot_bodies or rb in robot_bodies:
                    start = start + np.array([0, 0, -(dist + 1e-4)]); continue
                break
            vis += hit_straw
        sp = d.xpos[straw_bid]
        res["rows"].append(dict(f=f, door_state=round(float(val), 3), straw_pos=sp.round(3).tolist(), visible_frac=round(vis / len(pts), 3),
                                in_drawer_contact=bool(OU.check_obj_fixture_contact(env, "straw", dr))))
    print(json.dumps(res), flush=True)
