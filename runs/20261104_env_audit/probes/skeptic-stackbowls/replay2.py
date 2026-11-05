import json, sys, os, glob
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
sys.path.insert(0, f"{R}/runs/20261025_seen480")
os.chdir(f"{R}/runs/20261025_seen480")
from paths import local
import robocasa.utils.object_utils as OU
from analyze_stages import replay_env
import atomic_eval  # noqa
from oracle import reset_to, set_state

run = "run"
rows = {json.loads(l)["tag"]: json.loads(l) for p in glob.glob(f"{run}/results_w*.jsonl") for l in open(p)}
st = {json.loads(l)["tag"]: json.loads(l) for l in open("analysis/stages.jsonl")}
tags = sys.argv[1:]
env = replay_env("StackBowlsCabinet")
def preds(env):
    return dict(
        b1_in_cab=bool(OU.obj_inside_of(env, "bowl1", env.cabinet)),
        b2_in_cab=bool(OU.obj_inside_of(env, "bowl2", env.cabinet)),
        b1_in_b2=bool(OU.check_obj_in_receptacle(env, "bowl1", "bowl2")),
        b2_in_b1=bool(OU.check_obj_in_receptacle(env, "bowl2", "bowl1")),
        far1=bool(OU.gripper_obj_far(env, "bowl1")), far2=bool(OU.gripper_obj_far(env, "bowl2")),
        succ=bool(env._check_success()), touch1=bool(env.check_contact(env.robots[0].gripper["right"], env.objects["bowl1"])), touch2=bool(env.check_contact(env.robots[0].gripper["right"], env.objects["bowl2"])))
for tag in tags:
    tag = f"StackBowlsCabinet_{tag}_single"
    r = rows[tag]; s = st[tag]
    z = np.load(local(r["scene"]), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    states = np.load(f"{run}/StackBowlsCabinet/{tag}_steps.npz")["states"]
    f0 = s["stages"][0]["first_true"]; fr = s["labeler_debug"]["first_release"]
    out = {"tag": tag, "success": r["success"], "f0": f0, "nest_release": fr}
    for name, f in [("stage0_first", f0)]:
        if f is None or f >= len(states):
            continue
        set_state(env, states[f])
        out[name] = (f, preds(env))
    print(json.dumps(out), flush=True)
