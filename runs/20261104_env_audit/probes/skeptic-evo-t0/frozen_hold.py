"""On real frozen scene files: load_scene, then (a) OSC nullspace target initial_joint vs the arm qpos at t=0 and vs the
noise-free init_qpos, (b) hold still (zero action, base mode) for 16 steps from rollout()'s start (restore_node, forced obs)."""
import sys, json, numpy as np
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/mclee/robocasa")
import robosuite
_o = robosuite.make
robosuite.make = lambda *a, **kw: _o(*a, **{**kw, "use_camera_obs": False, "has_offscreen_renderer": False})
import gymnasium as gym, robocasa, robocasa.wrappers.gym_wrapper  # noqa
from atomic_eval import load_scene, restore_node
task = sys.argv[1]; paths = sys.argv[2:]
base = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; k = base.env
for p in paths:
    s0 = load_scene(k, p)
    restore_node(k, s0, None); k._get_observations(force_update=True)
    r = k.robots[0]; arm = r.composite_controller.part_controllers["right"]; eef = r.eef_site_id["right"]
    q0 = np.array(k.sim.data.qpos[r._ref_joint_pos_indexes]); p0 = np.array(k.sim.data.site_xpos[eef])
    row = dict(scene=p.split("/")[-1], ij_minus_qpos_t0=round(float(np.abs(arm.initial_joint - q0).max()), 4),
               ij_minus_noisefree=round(float(np.abs(arm.initial_joint - np.array(r.init_qpos)).max()), 6))
    for i in range(16):
        base.step({"action.end_effector_position": np.zeros(3), "action.end_effector_rotation": np.zeros(3),
                   "action.gripper_close": np.zeros(1), "action.base_motion": np.zeros(4), "action.control_mode": np.ones(1)})
    row["base_mode_hold16_eef_drift_mm"] = round(1000 * float(np.linalg.norm(k.sim.data.site_xpos[eef] - p0)), 2)
    print(json.dumps(row), flush=True)
