import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
tag = "SteamInMicrowave_s21_single"
row = [r for r in rows() if r["tag"] == tag][0]
env = make_env("SteamInMicrowave"); load_scene(env, scene_of(row))
states = np.load(f"{RUN}/SteamInMicrowave/{tag}_steps.npz")["states"]
g = env.robots[0].gripper["right"]
for f in range(len(states)):
    set_state(env, states[f]); s = env._check_success()
    if f >= 1320:
        start = env.check_contact(g, f"{env.microwave.name}_start_button"); stop = env.check_contact(g, f"{env.microwave.name}_stop_button")
        print(f, "on", env.microwave._turned_on, "succ", s, "start_contact", start, "stop_contact", stop, "door_open", env.microwave.is_open(env), "closed", env.microwave.is_closed(env=env))
