import os
import sys, json
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
import robocasa.utils.object_utils as OU
from atomic_eval import load_scene
SC = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scenes_seen5_v2"
for task, objs in (("ArrangeVegetables", ["vegetable1", "vegetable2"]), ("GatherVegetables", ["veg1", "veg2"]), ("CondimentCollection", ["condiment1", "condiment2"])):
    b = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped; e = b.env
    rows = []
    for s in range(20):
        load_scene(e, f"{SC}/{task}_s{s}.npz")
        d = dict(seed=s, names=[OU.get_obj_lang(e, obj_name=o) for o in objs], lang=e.get_ep_meta()["lang"])
        if "Condiment" in task:
            d["in_cab_at_start"] = [bool(OU.obj_inside_of(e, o, e.cab)) for o in objs]
            d["door"] = {k: round(float(v), 2) for k, v in e.cab.get_door_state(env=e).items()} if hasattr(e.cab, "get_door_state") else type(e.cab).__name__
            d["cab_type"] = type(e.cab).__name__
            d["on_counter"] = [bool(OU.check_obj_fixture_contact(e, o, e.counter)) for o in objs]
        else:
            d["in_sink"] = [bool(OU.obj_inside_of(e, o, e.sink)) for o in objs]
            d["on_board_at_start"] = [bool(OU.check_obj_in_receptacle(e, o, "cutting_board")) for o in objs]
            bp = e.sim.data.body_xpos[e.obj_body_id["cutting_board"]]; d["board_to_sink_xy"] = round(float(np.linalg.norm(bp[:2] - np.array(e.sink.pos[:2]))), 2)
            d["board_on_counter"] = bool(OU.check_obj_fixture_contact(e, "cutting_board", e.counter))
        d["success_at_start"] = bool(e._check_success())
        rows.append(d)
    print(task); [print("  ", json.dumps(r)) for r in rows[:4]]
    keys = [k for k in rows[0] if isinstance(rows[0][k], list) and isinstance(rows[0][k][0], bool)] + [k for k in rows[0] if isinstance(rows[0][k], bool)]
    print("   summary:", {k: sum(all(r[k]) if isinstance(r[k], list) else r[k] for r in rows) for k in keys}, "door types", sorted({json.dumps(r.get("door")) for r in rows})[:6] if "Condiment" in task else "")
