import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state
RD = f"{ROOT}/runs/20261014_asus_seen11"
env = robosuite.make("PackIdenticalLunches", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "PackIdenticalLunches" and "error" not in r}
Q = {15: ["meat0"], 17: ["vegetable0"], 27: ["vegetable0", "vegetable1", "GRIPPER"], 29: ["vegetable0"], 23: ["meat1"], 7: [], 35: []}


def rect_dist(f, p):
    c = np.array(f.pos)[:2]; yaw = float(f.rot) if np.ndim(f.rot) == 0 else 0.0
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    q = R @ (np.array(p[:2]) - c); hx, hy = f.size[0] / 2, f.size[1] / 2
    dx, dy = max(abs(q[0]) - hx, 0), max(abs(q[1]) - hy, 0)
    return float(np.hypot(dx, dy))


out = {}
for s in (7, 15, 17, 23, 27, 29):
    r = res[s]; z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    d = env.sim.data
    tp = {t: [round(float(x), 3) for x in d.body_xpos[env.obj_body_id[t]][:2]] for t in ("tupperware0", "tupperware1")}
    counters = {n: f for n, f in env.fixtures.items() if n.startswith("counter")}
    front = counters.get("counter_1_front_group")
    row = dict(tupperware_xy_start=tp)
    if s in Q and Q[s]:
        set_state(env, np.load(f"{RD}/PackIdenticalLunches/{r['tag']}_end.npz", allow_pickle=True)["state"])
        row["items"] = {}
        for o in Q[s]:
            p = d.site_xpos[env.robots[0].eef_site_id["right"]] if o == "GRIPPER" else d.body_xpos[env.obj_body_id[o]]
            dist = {n: rect_dist(f, p) for n, f in counters.items()}
            near = min(dist, key=dist.get)
            row["items"][o] = dict(xy=[round(float(x), 3) for x in p[:2]], z=round(float(p[2]), 2),
                                   to_front_counter_top_cm=round(100 * dist["counter_1_front_group"], 1) if front else None,
                                   nearest_counter=near, to_nearest_cm=round(100 * dist[near], 1))
    out[f"s{s}"] = row; print(s, json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
