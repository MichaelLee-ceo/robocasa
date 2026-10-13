import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
from oracle import reset_to
from demo_utils import load_demo
for task in ("StackBowlsCabinet", "SteamInMicrowave", "PreSoakPan"):
    b = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; k = b.env
    b.reset(seed=0); nat = {n: round(float(k.sim.model.body_mass[k.obj_body_id[n]]), 4) for n in k.objects}
    d = load_demo(task, 0); reset_to(k, d["xml"], d["ep_meta"]); k.unset_ep_meta()
    dem = {n: round(float(k.sim.model.body_mass[k.obj_body_id[n]]), 4) for n in k.objects}
    # inflated variant: MuJoCo-saved xml of the demo scene
    reset_to(k, k.sim.model.get_xml(), d["ep_meta"]); k.unset_ep_meta()
    inf = {n: round(float(k.sim.model.body_mass[k.obj_body_id[n]]), 4) for n in k.objects}
    print(task, "native(seed0)", nat, "| demo xml", dem, "| demo via MuJoCo-saved xml", inf)
    b.close()
