import os, sys, json
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
sys.path.insert(0, f"{R}/runs/20261025_seen480")
from analyze_stages import replay_env
from oracle import reset_to, set_state
from paths import local
import robocasa.utils.object_utils as OU
task, s, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tag = f"{task}_s{s}_single"
rows = {}
import glob
for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if r.get("tag") == tag and "error" not in r: rows = r
z = np.load(local(rows["scene"]), allow_pickle=True)
env = replay_env(task)
reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
st = np.load(f"{R}/runs/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
for f in range(len(st)):
    set_state(env, st[f])
    ok = env._check_success()
    if f < a: continue
    if task == "StoreLeftoversInBowl":
        c = dict(ch=OU.check_obj_in_receptacle(env, "chicken_drumstick", "bowl"), veg=OU.check_obj_in_receptacle(env, "vegetable", "bowl"),
                 fridge=env.fridge.check_rack_contact(env, "bowl"), far=OU.gripper_obj_far(env, "bowl"))
    elif task == "PackIdenticalLunches":
        c = {}
        for o in ["vegetable0", "vegetable1", "meat0", "meat1"]:
            c[o] = [int(OU.check_obj_in_receptacle(env, o, "tupperware0")), int(OU.check_obj_in_receptacle(env, o, "tupperware1")), int(OU.gripper_obj_far(env, o))]
    else:
        c = {}
    print(f, ok, c, flush=True)
    if f >= b: break
