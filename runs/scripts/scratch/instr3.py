import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import gymnasium as gym, robocasa, groot_eval
from atomic_eval import load_scene
from atomic_steps import TASKS
for t in ("ArrangeVegetables", "GatherVegetables", "CondimentCollection"):
    b = gym.make(f"robocasa/{t}", split="target", enable_render=False).unwrapped; e = b.env
    for s in (0, 1):
        load_scene(e, f"{os.path.expanduser(os.environ.get('ROBOCASA_ROOT', '~/Desktop/robocasa'))}/runs/scenes_seen5_v2/{t}_s{s}.npz"); S = {}
        print(t, s, "|", TASKS[t][0]["instr"](e, S), "|", TASKS[t][1]["instr"](e, S), "| checks", TASKS[t][0]["check"](e, S), TASKS[t][1]["check"](e, S))
