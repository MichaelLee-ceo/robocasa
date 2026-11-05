"""StirVegetables seen480 s7: live success at step 976, replay success_time ends at 4. Does the cross-episode
prev_obj_pos_* (not reset by _setup_scene) change the replayed count? usage: python -I stir_replay.py <seed>"""
import glob, json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa

seed = int(sys.argv[1])
task = "StirVegetables"
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
r = [x for x in rows if x["task"] == task and x["seed"] == seed and "error" not in x][0]
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
z = np.load(local(r["scene"]), allow_pickle=True)
states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
for variant in ("fresh", "stale_far", "stale_near0"):
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    for v in ("veg1", "veg2"):
        if hasattr(env, f"prev_obj_pos_{v}"):
            delattr(env, f"prev_obj_pos_{v}")
    if variant == "stale_far":
        env.prev_obj_pos_veg1 = np.array([5.0, 5.0, 1.0]); env.prev_obj_pos_veg2 = np.array([5.0, 5.0, 1.0])
    incs = []
    for f in range(len(states)):
        before = env.success_time
        set_state(env, states[f])
        if variant == "stale_near0" and f == 0:
            pass
        if env.success_time != before:
            incs.append(f)
    print(json.dumps(dict(tag=r["tag"], live_success=r["success"], live_steps=r["steps"], variant=variant,
                          success_time_end=env.success_time, increments_at=incs, check_success_end=bool(env._check_success()))), flush=True)
