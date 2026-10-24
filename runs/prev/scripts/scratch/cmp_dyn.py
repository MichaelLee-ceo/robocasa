import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
from atomic_eval import load_scene
from oracle import set_state
base = gym.make("robocasa/StackBowlsCabinet", split="target", enable_render=False).unwrapped; kenv = base.env
def roll(n=200):
    rng = np.random.default_rng(1); out = []
    for i in range(n):
        a = {"action.end_effector_position": np.array([0.3, 0.0, -0.2]) * np.sin(i / 20), "action.end_effector_rotation": np.zeros(3),
             "action.gripper_close": np.array([1.0 if i > 100 else -1.0]), "action.base_motion": np.zeros(4), "action.control_mode": np.array([0.0])}
        o, *_ = base.step(a); out.append(np.concatenate([np.ravel(o[k]) for k in sorted(o) if k.startswith("state.")]))
    return np.array(out)
base.reset(seed=0); st = kenv.sim.get_state().flatten().copy(); R1 = roll()
set_state(kenv, st); R2 = roll()
load_scene(kenv, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scenes_seen5/StackBowlsCabinet_s0.npz"); R3 = roll()
set_state(kenv, st); R4 = roll()
print("fresh vs set_state-only same env:", float(np.abs(R1 - R2).max()))
print("fresh vs load_scene:", float(np.abs(R1 - R3).max()))
print("load_scene vs set_state after load:", float(np.abs(R3 - R4).max()))
c = kenv.robots[0].composite_controller
print({k: v for k, v in vars(c).items() if isinstance(v, (int, float, str, bool))})
