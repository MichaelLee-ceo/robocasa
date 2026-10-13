import sys
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/Desktop/robocasa/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
import robosuite.utils.transform_utils as T
import robocasa.utils.env_utils as EnvUtils
from atomic_eval import load_scene
from oracle import set_state
b = gym.make("robocasa/SteamInMicrowave", split="target", enable_render=False).unwrapped; e = b.env
load_scene(e, "/home/michaellee/Desktop/robocasa/runs/scenes_seen5_v2/SteamInMicrowave_s1.npz")
def frame():
    bid = e.sim.model.body_name2id("mobilebase0_base"); p = e.sim.data.body_xpos[bid]; R = e.sim.data.body_xmat[bid].reshape(3, 3)
    return p, R
p, R = frame()
for name, fx in (("sink", e.sink), ("microwave", e.microwave)):
    v = R.T @ (np.array(fx.pos) - p); print("initial: robot-frame vector to", name, np.round(v, 2))
print("anchor ori", e.init_robot_base_ori_anchor, "sink-ref pose", EnvUtils.compute_robot_base_placement_pose(e, ref_fixture=e.sink), "mw pose", EnvUtils.compute_robot_base_placement_pose(e, ref_fixture=e.microwave))
print("mw pos", e.microwave.pos, "sink pos", e.sink.pos, "base", np.round(p, 2))
from robot_home import home_robot
OUT = "/home/michaellee/Desktop/robocasa/runs/20261013_scenes_v2"
rec = [x for x in np.load(f"{OUT}/SteamInMicrowave/SteamInMicrowave_s1_single_states.npz", allow_pickle=True)["recs"] if x["k"] == 3][0]
set_state(e, rec["state"]); p, R = frame(); print("handoff: vec to mw", np.round(R.T @ (np.array(e.microwave.pos) - p), 2))
print(home_robot(e, e.microwave)); p, R = frame(); print("homed: vec to mw", np.round(R.T @ (np.array(e.microwave.pos) - p), 2), "base", np.round(p, 2))
