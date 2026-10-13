import sys, os, json
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/Desktop/robocasa/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
import robocasa.utils.object_utils as OU
from atomic_eval import load_scene
from oracle import set_state
D = "/home/michaellee/Desktop/robocasa/runs/20261013_scenes_v2/StackBowlsCabinet"
b = gym.make("robocasa/StackBowlsCabinet", split="target", enable_render=False).unwrapped; k = b.env
for s in range(20):
    f = f"{D}/StackBowlsCabinet_s{s}_aseen_chain_end.npz"
    if not os.path.exists(f): print(s, "no end state"); continue
    load_scene(k, f"/home/michaellee/Desktop/robocasa/runs/scenes_seen5_v2/StackBowlsCabinet_s{s}.npz"); set_state(k, np.load(f, allow_pickle=True)["state"])
    c = dict(b1_cab=OU.obj_inside_of(k, "bowl1", k.cabinet), b2_cab=OU.obj_inside_of(k, "bowl2", k.cabinet),
             b2_in_b1=OU.check_obj_in_receptacle(k, "bowl2", "bowl1"), b1_in_b2=OU.check_obj_in_receptacle(k, "bowl1", "bowl2"),
             far1=OU.gripper_obj_far(k, obj_name="bowl1"), far2=OU.gripper_obj_far(k, obj_name="bowl2"))
    print(s, {kk: bool(v) for kk, v in c.items()})
