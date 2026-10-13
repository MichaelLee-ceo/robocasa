import sys, json
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts"); sys.path.insert(0, "/home/michaellee/Desktop/robocasa/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval, imageio.v3 as iio
from oracle import set_state
base = gym.make("robocasa/StackBowlsCabinet", split="target", enable_render=True).unwrapped; kenv = base.env
img = lambda: base.get_observation(kenv._get_observations(force_update=True))["video.robot0_agentview_left"].astype(float)
base.reset(seed=0); A = img(); xml = kenv.sim.model.get_xml(); meta = kenv.get_ep_meta(); st = kenv.sim.get_state().flatten()
# variant 1: reset_from_xml_string(xml) without edit_model_xml
kenv.set_ep_meta(meta); kenv.reset(); kenv.reset_from_xml_string(xml); kenv.sim.reset(); set_state(kenv, st); B = img()
# variant 2: with edit_model_xml
kenv.set_ep_meta(meta); kenv.reset(); kenv.reset_from_xml_string(kenv.edit_model_xml(xml)); kenv.sim.reset(); set_state(kenv, st); C = img()
# variant 3: just set_ep_meta + reset (scene rebuilt from ep_meta) + state
kenv.set_ep_meta(meta); kenv.reset(); set_state(kenv, st); D = img()
kenv.unset_ep_meta()
for n, X in (("raw xml", B), ("edit_model_xml", C), ("ep_meta only", D)):
    print(n, "mean", round(float(np.abs(A - X).mean()), 3), "max", float(np.abs(A - X).max()), "frac px >10:", round(float((np.abs(A - X).max(-1) > 10).mean()), 4))
iio.imwrite("/tmp/claude-1001/-home-michaellee-Desktop-robocasa/d2b6537f-619c-5af7-ab37-a54e14a4b87a/scratchpad/render_cmp.png", np.concatenate([A, C, np.clip(np.abs(A - C) * 4, 0, 255)], 1).astype(np.uint8))
