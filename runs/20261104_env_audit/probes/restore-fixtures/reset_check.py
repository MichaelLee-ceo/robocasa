"""Does load_scene give fresh python state (fixtures + task attrs) when the worker's env ran an earlier episode?"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
task = "StirVegetables"
R = {r["tag"]: r for r in rows() if r["task"] == task}
env = make_env(task)
load_scene(env, scene_of(R["StirVegetables_s21_single"]))
fx_ids = {n: id(f) for n, f in env.fixtures.items()}
env.success_time = 99; env.prev_obj_pos_veg1 = np.array([9., 9., 9.]); mw = [f for f in env.fixtures.values() if type(f).__name__ == "Microwave"][0]; mw._turned_on = True
load_scene(env, scene_of(R["StirVegetables_s4_single"]))
print(json.dumps(dict(success_time=env.success_time, prev_obj_pos_veg1=getattr(env, "prev_obj_pos_veg1", "<unset>").tolist()
                      if hasattr(env, "prev_obj_pos_veg1") else "<unset>", microwave_on=[f for f in env.fixtures.values() if type(f).__name__ == "Microwave"][0]._turned_on,
                      fixtures_new_objects=all(id(f) != fx_ids.get(n) for n, f in env.fixtures.items()))))
