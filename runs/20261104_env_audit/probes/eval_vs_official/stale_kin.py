"""After a live env.step, are mjData kinematics/contacts (what _check_success reads) those of the final qpos?
Compare eef site pos and contact set before and after sim.forward()."""
import sys, numpy as np
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/mclee/robocasa")
import robosuite
_o = robosuite.make
robosuite.make = lambda *a, **kw: _o(*a, **{**kw, "use_camera_obs": False, "has_offscreen_renderer": False})
import gymnasium as gym, robocasa, robocasa.wrappers.gym_wrapper  # noqa
base = gym.make("robocasa/KettleBoiling", split="target", enable_render=False).unwrapped; k = base.env
base.reset(seed=0); r = np.random.default_rng(0)
eef = k.robots[0].eef_site_id["right"]
for i in range(12):
    a = {"action.end_effector_position": r.uniform(-1, 1, 3), "action.end_effector_rotation": r.uniform(-.5, .5, 3),
         "action.gripper_close": np.array([float(i > 6)]), "action.base_motion": np.zeros(4), "action.control_mode": np.array([0.0])}
    base.step(a)
    p0 = np.array(k.sim.data.site_xpos[eef]); c0 = sorted((int(c.geom1), int(c.geom2)) for c in k.sim.data.contact[:k.sim.data.ncon])
    k.sim.forward()
    p1 = np.array(k.sim.data.site_xpos[eef]); c1 = sorted((int(c.geom1), int(c.geom2)) for c in k.sim.data.contact[:k.sim.data.ncon])
    print(i, "eef shift by forward (mm):", round(1000 * float(np.linalg.norm(p1 - p0)), 3), "contacts same:", c0 == c1, len(c0), len(c1))
