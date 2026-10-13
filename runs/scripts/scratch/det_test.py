import sys; sys.path.insert(0, "/home/michaellee/Desktop/robocasa/runs/scripts")
sys.path.insert(0, "/home/michaellee/Desktop/robocasa/third_party/Isaac-GR00T")
import groot_eval
import gymnasium as gym, robocasa, numpy as np, json, sys

def fp(kenv):
    m = kenv.get_ep_meta()
    objs = [(c["name"], c.get("info", {}).get("mjcf_path", "")[-60:]) for c in m["object_cfgs"]]
    return dict(layout=m.get("layout_id"), style=m.get("style_id"), objs=objs, nq=kenv.sim.model.nq)
e = gym.make("robocasa/SteamInMicrowave", split="target", enable_render=False).unwrapped
out = []
for s in [5, 3, 5, 0, 5]:
    e.reset(seed=s); out.append((s, fp(e.env)))
for s, f in out: print(s, f["layout"], f["style"], f["nq"], [o[1].split('/')[-2] if '/' in o[1] else o[1] for o in f["objs"]])
