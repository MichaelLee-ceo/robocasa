"""Independent check: at the logged stage-pass steps of seen480 PIL episodes, evaluate OUR stage checks (fresh S),
the official _check_success, and the per-tupperware composition; plus a proposed 'no dead end' predicate."""
import glob, json, sys, copy
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
import robocasa.utils.object_utils as OU  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa
from atomic_steps import TASKS  # noqa

VEG, MEAT = ["vegetable0", "vegetable1"], ["meat0", "meat1"]
task = "PackIdenticalLunches"
steps = TASKS[task]
keep = [int(x) for x in sys.argv[1].split(",")]
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
rows = {r["seed"]: r for r in rows if r["task"] == task and "error" not in r}
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()

def comp(e):
    out = {tw: [o for o in VEG + MEAT if OU.check_obj_in_receptacle(e, o, tw)] for tw in ("tupperware0", "tupperware1")}
    both = sorted(set(out["tupperware0"]) & set(out["tupperware1"]))
    dup = any(sum(o in VEG for o in v) > 1 or sum(o in MEAT for o in v) > 1 for v in out.values())
    return out, both, bool(dup or both)

for s in keep:
    r = rows[s]
    z = np.load(local(r["scene"]), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
    print(r["tag"], "success", r["success"], "reached", r["reached"], "history", r["history"])
    ts = [t for t, k in r["history"]] + [len(states) - 1]
    for t in ts:
        set_state(env, states[t])
        out, both, dead = comp(env)
        S = {}
        stage = [bool(steps[q]["check"](env, copy.deepcopy(S))) for q in range(4)]
        print(f"  t={t:5d} our stage checks 0..3 (fresh S) {stage} official {bool(env._check_success())} "
              f"t0={out['tupperware0']} t1={out['tupperware1']} both={both} dead_end={dead}")
