"""Scene = first reset(seed) of a freshly created env (no xml round-trip: recompiling model.get_xml() changes body
masses/inertias). Each (task, seed) is generated in its own new process; only the state vector and a fingerprint are kept."""
import os, sys, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Desktop/robocasa/third_party/Isaac-GR00T"))
import gymnasium as gym
import robocasa  # noqa
import groot_eval  # noqa
task, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
base = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; kenv = base.env
base.reset(seed=seed)
m = kenv.get_ep_meta()
fp = dict(layout=m.get("layout_id"), style=m.get("style_id"), nq=int(kenv.sim.model.nq),
          objects={c["name"]: c.get("info", {}).get("mjcf_path", "").split("/objects/")[-1] for c in m["object_cfgs"]},
          mass_sha1=hashlib.sha1(np.round(kenv.sim.model.body_mass, 6).tobytes()).hexdigest()[:12])
st = kenv.sim.get_state().flatten()
np.savez_compressed(out, state=st, fp=json.dumps(fp), lang=m["lang"])
print(json.dumps(dict(task=task, seed=seed, state_sha1=hashlib.sha1(np.round(st, 8).tobytes()).hexdigest()[:12], **fp)), flush=True)
