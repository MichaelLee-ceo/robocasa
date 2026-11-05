"""StirVegetables seen480: at the boundary where our stage 2 (PickUpSpatula) passed and at the end, are the other
conjuncts that the official stir counter needs (pot within 0.15 m of the task burner with the knob on, both vegetables
in the pot) still true? Our stages 0-2 do not check the pot's location."""
import glob, json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
import robocasa.utils.object_utils as OU  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa

task = "StirVegetables"
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
rows = sorted([r for r in rows if r["task"] == task and "error" not in r], key=lambda r: r["seed"])
if len(sys.argv) > 1:
    keep = {int(x) for x in sys.argv[1].split(",")}
    rows = [r for r in rows if r["seed"] in keep]
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()


def conj(e):
    loc = e.stove.check_obj_location_on_stove(e, "pot", threshold=0.15)
    return dict(pot_loc_ok=bool(loc == e.knob), pot_loc=loc, veg1=bool(OU.check_obj_in_receptacle(e, "veg1", "pot")),
                veg2=bool(OU.check_obj_in_receptacle(e, "veg2", "pot")))


for r in rows:
    z = np.load(local(r["scene"]), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
    at = {}
    for t, k in r["history"]:
        set_state(env, states[t])
        at[k] = dict(t=t, **conj(env))
    set_state(env, states[-1])
    print(json.dumps(dict(tag=r["tag"], success=r["success"], reached=r["reached"], at_stage_pass=at, end=conj(env))), flush=True)
