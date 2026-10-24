import gymnasium as gym, robocasa, json, numpy as np
e = gym.make("robocasa/CuttingToolSelection", split="target", enable_render=True).unwrapped
out=[]
for s in [0,0,1]:
    e.reset(seed=s); m=e.env.get_ep_meta()
    out.append((m.get("layout_id"), m.get("style_id"), sorted((k, v.get("info",{}).get("mjcf_path","")[-40:]) for k,v in m.get("object_cfgs",[{}])[0].items()) if False else [ (c.get("name"), c.get("info",{}).get("mjcf_path","")[-30:]) for c in m.get("object_cfgs",[])], float(e.env.sim.data.qpos[:5].sum())))
for o in out: print(o)
print("seed0 deterministic:", out[0]==out[1])
