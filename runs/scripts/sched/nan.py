from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES
import robocasa.models.objects.kitchen_object_utils as K
import numpy as np, gymnasium as gym, robocasa
orig = np.random.Generator.choice
for task in ["PortionHotDogs","GarnishPancake"]:
    try:
        gym.make(f"robocasa/{task}", split="target", enable_render=False)
        print(task, "ok")
    except Exception as e:
        import traceback; tb=traceback.extract_tb(e.__traceback__)
        fr=[f for f in tb if f.name=="sample_kitchen_object_helper"]
        print(task, "fail", repr(e))
# list categories with no objaverse/lightwheel objects
empty=[c for c,v in OBJ_CATEGORIES.items() if not any(r in v and len(v[r].mjcf_paths)>0 for r in ("objaverse","lightwheel"))]
print("categories lacking objaverse+lightwheel:", empty[:60])
print({c: list(OBJ_CATEGORIES[c].keys()) for c in OBJ_CATEGORIES if 'hot' in c or 'pancake' in c or 'bun' in c or 'sausage' in c})
