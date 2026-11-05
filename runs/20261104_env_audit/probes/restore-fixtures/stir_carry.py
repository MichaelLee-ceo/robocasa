"""StirVegetables: replay with prev_obj_pos_* unset (fresh process) vs left over from another episode (far away)."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
tag = sys.argv[1]
row = [r for r in rows() if r["tag"] == tag][0]
env = make_env("StirVegetables")
states = np.load(f"{RUN}/StirVegetables/{tag}_steps.npz")["states"]
out = dict(tag=tag, online_success=row["success"])
for mode in ("fresh", "leftover"):
    load_scene(env, scene_of(row))
    for n in ("veg1", "veg2"):
        if hasattr(env, f"prev_obj_pos_{n}"):
            delattr(env, f"prev_obj_pos_{n}")
        if mode == "leftover":
            setattr(env, f"prev_obj_pos_{n}", np.array([5.0, 5.0, 1.0]))
    first = None
    for f in range(len(states)):
        set_state(env, states[f])
        if env.success_time and first is None:
            first = f
    out[mode] = dict(success_time=env.success_time, first_count_frame=first)
print(json.dumps(out))
