import sys, time; sys.path.insert(0, sys.argv[1])
from harness import *
from atomic_eval import load_scene
env = make_env("KettleBoiling")
s = load_scene(env, scene_path("KettleBoiling", 0))
m = env.sim.model
for b in range(m.nbody):
    if m.body_mocapid[b] >= 0: print("mocap body", m.body_id2name(b), m.body_mocapid[b], "ngeoms", sum(m.geom_bodyid==b))
print(env.sim.data.mocap_pos, env.sim.data.mocap_quat)
