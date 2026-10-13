import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
import robocasa.utils.env_utils as EnvUtils
from atomic_eval import load_scene
from oracle import set_state
OUT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/20261013_scenes_v2"
b = gym.make("robocasa/SteamInMicrowave", split="target", enable_render=False).unwrapped; e = b.env
load_scene(e, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scenes_seen5_v2/SteamInMicrowave_s1.npz")
rec = [x for x in np.load(f"{OUT}/SteamInMicrowave/SteamInMicrowave_s1_single_states.npz", allow_pickle=True)["recs"] if x["k"] == 3][0]
set_state(e, rec["state"]); e.robot_geom_ids = None
e.robots[0].reset(deterministic=True)
pos, ori = EnvUtils.compute_robot_base_placement_pose(e, ref_fixture=e.microwave)
m = e.sim.model
e.sim.data.qpos[m.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")] = np.pi; e.sim.forward()
EnvUtils.set_robot_to_position(e, pos); e.sim.forward()
EnvUtils.detect_robot_collision(e)
for i in range(e.sim.data.ncon):
    c = e.sim.data.contact[i]; g1, g2 = c.geom1, c.geom2
    if (g1 in e.robot_geom_ids) != (g2 in e.robot_geom_ids):
        print(m.geom_id2name(g1), "|", m.geom_id2name(g2), round(float(c.dist), 4))
