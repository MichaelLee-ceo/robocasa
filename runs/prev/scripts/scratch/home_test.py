import os
import sys, json
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval, imageio.v3 as iio
from atomic_eval import load_scene
from oracle import set_state
from robot_home import home_robot
OUT = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/20261013_scenes_v2"
rows = []
for task, k, ref, seeds in (("SteamInMicrowave", 3, "microwave", [1, 2]), ("PreSoakPan", 2, "sink", [0, 6])):
    b = gym.make(f"robocasa/{task}", split="target", enable_render=True).unwrapped; e = b.env
    for s in seeds:
        rec = [x for x in np.load(f"{OUT}/{task}/{task}_s{s}_single_states.npz", allow_pickle=True)["recs"] if x["k"] == k][0]
        load_scene(e, f"{os.path.expanduser(os.environ.get('ROBOCASA_ROOT', '~/Desktop/robocasa'))}/runs/scenes_seen5_v2/{task}_s{s}.npz"); set_state(e, rec["state"])
        img = lambda: np.concatenate([b.get_observation(e._get_observations(force_update=True))[c] for c in ("video.robot0_agentview_left", "video.robot0_eye_in_hand")], 1)
        before = img(); info = home_robot(e, getattr(e, ref)); after = img()
        print(task, s, info, "door/water:", e.microwave.is_closed(env=e) if ref == "microwave" else e.sink.get_handle_state(env=e)["water_on"])
        rows.append(np.concatenate([before, after], 1))
iio.imwrite("/tmp/claude-1001/-home-michaellee-Desktop-robocasa/d2b6537f-619c-5af7-ab37-a54e14a4b87a/scratchpad/home_test.jpg", np.concatenate(rows, 0)[::2, ::2])
