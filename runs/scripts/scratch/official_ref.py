import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts")
import groot_eval  # noqa
for task, ref in (("TurnOnMicrowave", "microwave"), ("TurnOnSinkFaucet", "sink")):
    b = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; e = b.env
    for s in range(4):
        b.reset(seed=s); bid = e.sim.model.body_name2id("mobilebase0_base")
        p = e.sim.data.body_xpos[bid]; R = e.sim.data.body_xmat[bid].reshape(3, 3)
        print(task, s, "ref in robot frame", np.round(R.T @ (np.array(getattr(e, ref).pos) - p), 2))
