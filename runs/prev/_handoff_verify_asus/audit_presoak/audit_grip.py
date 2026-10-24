"""Gripper-object distances and each PreSoakPan tracker-step conjunct at the END state of given seeds.
usage: python -I audit_grip.py <seed> [<seed> ...]"""
import glob, json, os, sys
import numpy as np, gymnasium as gym
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa, groot_eval  # noqa
from oracle import reset_to, set_state
import robocasa.utils.object_utils as OU
RUN = f"{ROOT}/runs/20261014_asus_seen5"
R = {}
for p in glob.glob(f"{RUN}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l); R[r["seed"]] = r
e = gym.make("robocasa/PreSoakPan", split="target", enable_render=False).unwrapped.env
for s in map(int, sys.argv[1:]):
    z = np.load(R[s]["scene"], allow_pickle=True)
    reset_to(e, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); e.unset_ep_meta()
    set_state(e, np.load(f"{RUN}/PreSoakPan/PreSoakPan_s{s}_single_end.npz", allow_pickle=True)["state"])
    g = e.sim.data.site_xpos[e.robots[0].eef_site_id["right"]]
    out = dict(seed=s)
    for n in ("obj1", "obj2"):
        out[n] = dict(in_sink=bool(OU.obj_inside_of(e, n, e.sink, partial_check=False)),
                      grip_dist=round(float(np.linalg.norm(g - e.sim.data.body_xpos[e.obj_body_id[n]])), 3),
                      grip_far=bool(OU.gripper_obj_far(e, n)))
    out["water_on"] = bool(e.sink.get_handle_state(env=e)["water_on"])
    print(json.dumps(out), flush=True)
