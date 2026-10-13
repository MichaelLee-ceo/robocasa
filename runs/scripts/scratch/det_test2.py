import sys; sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts")
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/third_party/Isaac-GR00T")
import groot_eval
import gymnasium as gym, robocasa, numpy as np, json, sys
import gymnasium as gym, robocasa, numpy as np
def fp(e):
    m = e.env.get_ep_meta(); return (m.get("layout_id"), m.get("style_id"), e.env.sim.model.nq, tuple(c.get("info", {}).get("mjcf_path", "").split("/")[-2] for c in m["object_cfgs"]))
e = gym.make("robocasa/SteamInMicrowave", split="target", enable_render=False).unwrapped
e.reset(seed=3); print("after 3 no steps ->", end=" "); e.reset(seed=5); print(fp(e))
e.reset(seed=3)
for _ in range(200): e.step(e.action_space.sample())
e.reset(seed=5); print("after 3 + 200 random steps ->", fp(e))
e2 = gym.make("robocasa/SteamInMicrowave", split="target", enable_render=False).unwrapped
e2.reset(seed=5); print("fresh env ->", fp(e2))
e2.reset(seed=5); print("fresh env, 5 twice ->", fp(e2))
