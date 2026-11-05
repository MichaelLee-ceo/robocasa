"""Official reset(seed) vs frozen-scene start (reset_to + set_state + forced obs, as load_scene + rollout(hidden=None)).
Apply a ZERO action (hold still) for 16 steps in base mode, then 16 in arm mode; report controller state and drift."""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/mclee/robocasa")
import robosuite
_o = robosuite.make
robosuite.make = lambda *a, **kw: _o(*a, **{**kw, "use_camera_obs": False, "has_offscreen_renderer": False})
import gymnasium as gym, robocasa, robocasa.wrappers.gym_wrapper  # noqa
from oracle import reset_to, set_state

task, seed, order = sys.argv[1], int(sys.argv[2]), sys.argv[3]  # order: "base_first" or "arm_first"
base = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; k = base.env
r = k.robots[0]
eef = r.eef_site_id["right"]
def arm_c():
    return k.robots[0].composite_controller.part_controllers["right"]
def zero(mode):
    return {"action.end_effector_position": np.zeros(3, np.float32), "action.end_effector_rotation": np.zeros(3, np.float32),
            "action.gripper_close": np.zeros(1, np.float32), "action.base_motion": np.zeros(4, np.float32),
            "action.control_mode": np.array([mode], np.float32)}
modes = ([1.0] * 16 + [0.0] * 16) if order == "base_first" else ([0.0] * 96)
def ctrl():
    arm = arm_c()
    return dict(initial_joint=np.array(arm.initial_joint), goal_pos=np.array(arm.goal_pos), goal_ori=np.array(arm.goal_ori))
def run():
    p0 = np.array(k.sim.data.site_xpos[eef]); q0 = np.array(k.sim.data.qpos[r._ref_joint_pos_indexes])
    out = []
    for i, m in enumerate(modes):
        base.step(zero(m))
        if i in (0, 3, 15, 16, 31, 63, 95):
            out.append(dict(step=i, mode=m, eef_drift_mm=round(1000 * float(np.linalg.norm(k.sim.data.site_xpos[eef] - p0)), 2),
                            arm_q_drift_rad=round(float(np.abs(k.sim.data.qpos[r._ref_joint_pos_indexes] - q0).max()), 4)))
    return out
base.reset(seed=seed)
xml, meta, s0 = k.model.get_xml(), k.get_ep_meta(), k.sim.get_state().flatten().copy()
init_noisefree = np.array(r.init_qpos)
r = k.robots[0]; cA = ctrl(); qA = np.array(k.sim.data.qpos[r._ref_joint_pos_indexes])
outA = run()
reset_to(k, xml, meta, s0); k.unset_ep_meta(); set_state(k, s0)
k._get_observations(force_update=True)
assert np.array_equal(k.sim.get_state().flatten(), s0)
r = k.robots[0]; cB = ctrl()

outB = run()
print(json.dumps(dict(task=task, seed=seed, order=order,
    official_initial_joint_minus_qpos_t0=round(float(np.abs(cA["initial_joint"] - qA).max()), 4),
    arm_qpos_t0_minus_noisefree_init=round(float(np.abs(qA - init_noisefree).max()), 4),
    official_initial_joint_minus_noisefree=round(float(np.abs(cA["initial_joint"] - init_noisefree).max()), 4),
    frozen_initial_joint_minus_noisefree=round(float(np.abs(cB["initial_joint"] - init_noisefree).max()), 6),
    initial_joint_diff=round(float(np.abs(cA["initial_joint"] - cB["initial_joint"]).max()), 4),
    goal_pos_diff_mm=round(1000 * float(np.linalg.norm(cA["goal_pos"] - cB["goal_pos"])), 2),
    goal_ori_diff=round(float(np.abs(cA["goal_ori"] - cB["goal_ori"]).max()), 4),
    official=outA, frozen=outB)), flush=True)
