import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
from oracle import reset_to
task = sys.argv[1]
def model_arrays(m):
    out = {}
    for f in dir(m):
        if f.startswith("_"): continue
        try: v = getattr(m, f)
        except Exception: continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiu": out[f] = v.copy()
    return out
b = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; k = b.env
b.reset(seed=3); ref = model_arrays(k.sim.model._model); st = k.sim.get_state().flatten().copy(); meta = k.get_ep_meta()
xml_py = k.model.get_xml()
reset_to(k, xml_py, meta, st); k.unset_ep_meta(); new = model_arrays(k.sim.model._model)
bad = [(f, float(np.abs(ref[f].astype(float) - new[f].astype(float)).max())) for f in ref if f in new and ref[f].shape == new[f].shape and ref[f].size and np.abs(ref[f].astype(float) - new[f].astype(float)).max() > 1e-6]
shape = [f for f in ref if f in new and ref[f].shape != new[f].shape]
print(task, "model.get_xml() reload: differing arrays", bad[:12], "shape-diff", shape[:5])
print("state equal", float(np.abs(k.sim.get_state().flatten() - st).max()))
