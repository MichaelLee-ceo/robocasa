import sys, os
import numpy as np
R = "/home/michaellee/mclee/robocasa/runs"
sys.path[:0] = [f"{R}/20261028_critical", f"{R}/scripts", os.path.dirname(os.path.abspath(__file__))]
import mkenv, data
from atomic_eval import load_scene
from oracle import set_state
import robocasa.utils.object_utils as OU
env = mkenv.make("SteamInMicrowave")
for tag, ts in (("SteamInMicrowave_s6_single", (304, 400, 432, 688, 800, 1400)), ("SteamInMicrowave_s28_single", (272, 640, 672, 816, 1072, 1177, 1200, 1400))):
    load_scene(env, data.scenes()[tag])
    st = np.load(f"{R}/20261025_seen480/run/SteamInMicrowave/{tag}_steps.npz")["states"]
    for t in ts:
        t = min(t, len(st) - 1)
        set_state(env, st[t])
        p = env.sim.data.body_xpos[env.obj_body_id["vegetable"]]
        print(tag, t, "veg_in_bowl", OU.check_obj_in_receptacle(env, "vegetable", "bowl"), "bowl_in_mw", OU.obj_inside_of(env, "bowl", env.microwave),
              "door_closed", env.microwave.is_closed(env=env), "turned_on", env.microwave.get_state()["turned_on"], "veg z %.3f" % p[2])
