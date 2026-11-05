"""Constructed live-physics test of _toast_done vs the official latch (CPU, no rendering).
Case L (laxer): bread seated at a chunk boundary (bread_slot latched), then moved onto the plate without a grasp,
then the bread's lever switched on; step to the pop-up. Expect _toast_done True, env.toaster_on False.
Case S (stuck): bread stays seated, lever switched on; after the 500-step cycle the lever is held at 1.0 (emulating
the s27 jam). Expect env.toaster_on True, _toast_done False."""
import json, sys, copy
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts"); sys.path.insert(0, f"{R}/runs/20261025_seen480")
from analyze_stages import replay_env
import atomic_eval  # noqa
from atomic_eval import load_scene
from atomic_steps_seen11 import _toast_done
import robocasa.utils.object_utils as OU

scene = sys.argv[1]; case = sys.argv[2]
env = replay_env("GetToastedBread")
load_scene(env, f"{R}/runs/scenes_seen11_v2/{scene}.npz")
env.toaster_on = False
adim = env.action_dim
a0 = np.zeros(adim)
print("action_dim", adim, "pairs", list(env.toaster.get_state(env).keys()))
S = {}
def boundary(t):
    r = _toast_done(env, S)
    return r
t = 0
for _ in range(16):
    env.step(a0); t += 1
print("t", t, "_toast_done", boundary(t), "S", S, "contact", [env.toaster.check_slot_contact(env, "obj", slot_pair=i) for i in range(len(env.toaster.get_state(env)))])
bs = S.get("bread_slot")
if case == "L":
    # move the bread onto the plate (no grasp)
    jn = env.objects["obj"].joints[0]
    adr = env.sim.model.get_joint_qpos_addr(jn)
    q = env.sim.data.qpos[adr[0]:adr[1]].copy()
    pp = env.sim.data.body_xpos[env.obj_body_id["plate"]].copy()
    q[:3] = pp + np.array([0, 0, 0.06])
    env.sim.data.qpos[adr[0]:adr[1]] = q
    env.sim.data.qvel[env.sim.model.get_joint_qvel_addr(jn)[0]:env.sim.model.get_joint_qvel_addr(jn)[1]] = 0
    env.sim.forward()
    for _ in range(16):
        env.step(a0); t += 1
    print("after move: grasped", OU.check_obj_grasped(env, "obj"), "in plate", OU.check_obj_in_receptacle(env, "obj", "plate"),
          "slot contact", env.toaster.check_slot_contact(env, "obj", slot_pair=bs), "_toast_done", boundary(t), "S", S)
env.toaster.set_lever(env, bs, 1.0)
first_done = None; on_seen = False
for _ in range(1200):
    env.step(a0); t += 1
    st = env.toaster.get_state(env, slot_pair=bs)
    on_seen |= bool(st["turned_on"])
    if case == "S" and on_seen and not st["turned_on"]:
        env.toaster.set_lever(env, bs, 1.0)  # emulate the jam: lever stays down after the cycle
    if t % 16 == 0:
        d = boundary(t)
        if d and first_done is None:
            first_done = t
    if t % 160 == 0:
        print(t, "on", st["turned_on"], "lever", round(float(st["lever"]), 3), "toaster_on", env.toaster_on, "S", S, "_toast_done", _toast_done(env, copy.deepcopy(S)))
print("RESULT case", case, "first _toast_done", first_done, "env.toaster_on", env.toaster_on, "_check_success", env._check_success(),
      "in plate", OU.check_obj_in_receptacle(env, "obj", "plate"), "far", OU.gripper_obj_far(env, "obj"))
