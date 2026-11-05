"""PackIdenticalLunches (seen480 plain runs): composition of the two tupperwares at every chunk boundary where our
stage tracker advanced, and at the end. A configuration is a dead end for our stage sequence when a tupperware holds
two vegetables or two meats, or an object counts in both tupperwares (official _check_success needs exactly one of
each per tupperware and no duplicates) -- reaching it means a placed object must be taken out again."""
import glob, json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
sys.path.insert(0, f"{R}/third_party/Isaac-GR00T")
import robosuite  # noqa
import robocasa.utils.object_utils as OU  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa

VEG, MEAT = ["vegetable0", "vegetable1"], ["meat0", "meat1"]


def comp(e):
    out = {}
    for tw in ("tupperware0", "tupperware1"):
        out[tw] = [o for o in VEG + MEAT if OU.check_obj_in_receptacle(e, o, tw)]
    both = sorted(set(out["tupperware0"]) & set(out["tupperware1"]))
    dup = any(sum(o in VEG for o in v) > 1 or sum(o in MEAT for o in v) > 1 for v in out.values())
    return dict(t0=out["tupperware0"], t1=out["tupperware1"], both=both, dead=bool(dup or both))


task = sys.argv[1] if len(sys.argv) > 1 else "PackIdenticalLunches"
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
rows = sorted([r for r in rows if r["task"] == task and "error" not in r], key=lambda r: r["seed"])
if len(sys.argv) > 2:
    keep = {int(x) for x in sys.argv[2].split(",")}
    rows = [r for r in rows if r["seed"] in keep]
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
summary = []
for r in rows:
    z = np.load(local(r["scene"]), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
    at = []
    for t, k in r["history"]:
        set_state(env, states[t])
        at.append(dict(t=t, stage_done=k, **comp(env)))
    # first chunk boundary at which the configuration is a dead end
    first_dead = None
    for t in range(16, len(states), 16):
        set_state(env, states[t])
        if comp(env)["dead"]:
            first_dead = t
            break
    set_state(env, states[-1])
    end = comp(env)
    rec = dict(tag=r["tag"], success=r["success"], reached=r["reached"], steps=r["steps"], at_stage=at, end=end,
               first_dead_boundary=first_dead)
    summary.append(rec)
    print(json.dumps(rec), flush=True)
print("SUMMARY", json.dumps(dict(
    n=len(summary),
    dead_at_some_stage_pass=[s["tag"] for s in summary if any(a["dead"] for a in s["at_stage"])],
    dead_at_end=[s["tag"] for s in summary if s["end"]["dead"]],
    ever_dead=[(s["tag"], s["first_dead_boundary"], s["reached"], s["success"]) for s in summary if s["first_dead_boundary"] is not None])))
