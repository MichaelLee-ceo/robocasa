import sys, os, json
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/Desktop/robocasa/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
from atomic_eval import load_scene
from oracle import set_state
from robot_home import home_robot
OUT = "/home/michaellee/Desktop/robocasa/runs/20261013_scenes_v2"
for task, k, ref in (("SteamInMicrowave", 3, "microwave"), ("PreSoakPan", 2, "sink")):
    b = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; e = b.env
    for s in range(20):
        recs = [x for x in np.load(f"{OUT}/{task}/{task}_s{s}_single_states.npz", allow_pickle=True)["recs"] if x["k"] == k]
        if not recs: continue
        load_scene(e, f"/home/michaellee/Desktop/robocasa/runs/scenes_seen5_v2/{task}_s{s}.npz"); set_state(e, recs[0]["state"])
        bid = e.sim.model.body_name2id("mobilebase0_base"); p = e.sim.data.body_xpos[bid].copy(); R = e.sim.data.body_xmat[bid].reshape(3, 3).copy()
        before = np.round(R.T @ (np.array(getattr(e, ref).pos) - p), 2).tolist()
        info = home_robot(e, getattr(e, ref))
        print(json.dumps(dict(task=task, seed=s, handoff_ref_in_robot=before, **info)), flush=True)
