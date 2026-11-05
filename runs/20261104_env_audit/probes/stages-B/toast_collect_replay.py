"""GetToastedBread seen480 episodes whose toasting finished (our stage 0 passed): after replaying the saved states the
way runs/20261026_critic_data/collect.py replay() does (set_state per step, stage checks at chunk boundaries, no
_check_success per step), is the official latch env.toaster_on set? And with _check_success per step (stage_trace)?
usage: python -I toast_collect_replay.py <seed> [<seed> ...]"""
import glob, json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa
from atomic_steps import TASKS  # noqa

task = "GetToastedBread"
steps = TASKS[task]
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
for seed in map(int, sys.argv[1:]):
    r = [x for x in rows if x["task"] == task and x["seed"] == seed and "error" not in x][0]
    z = np.load(local(r["scene"]), allow_pickle=True)
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
    t0 = r["history"][0][0] if r["history"] else len(states) - 1
    out = dict(tag=r["tag"], live_reached=r["reached"], stage0_logged_at=t0)
    for mode in ("collect_replay", "per_step_check_success"):
        reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
        S, k = {}, 0
        for i in range(t0 + 1):
            set_state(env, states[i])
            if mode == "per_step_check_success":
                env._check_success()
            if i and i % 16 == 0:
                while k < len(steps) and steps[k]["check"](env, S):
                    k += 1
        out[mode] = dict(k=k, S=S, env_toaster_on=bool(env.toaster_on))
    print(json.dumps(out), flush=True)
