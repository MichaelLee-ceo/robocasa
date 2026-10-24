"""Per KettleBoiling scene: stove geometry group, burner nearest the kettle at the end, knobs ON, knob left-to-right
indices; plus knob orders of stove groups in SearingMeat. Read-only, CPU only."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state


def geom(env):
    st, m, d = env.stove, env.sim.model, env.sim.data
    bid = m.body_name2id(st.root_body); Rm = d.body_xmat[bid].reshape(3, 3); c = d.body_xpos[bid]
    L = lambda p: Rm.T @ (np.array(p) - c)
    burners = {k: np.array(d.get_site_xpos(v.get("name"))) for k, v in st.burner_sites.items() if v is not None}
    knobs = {k: np.array(d.xanchor[m.joint_name2id(f"{st.naming_prefix}knob_{k}_joint")]) for k, j in st.knob_joints.items() if j is not None}
    order = sorted(knobs, key=lambda k: L(knobs[k])[0])
    key = tuple(sorted((k, tuple(np.round(100 * L(v)[:2], 0))) for k, v in list(burners.items()) + [("knob_" + k, v) for k, v in knobs.items()]))
    return burners, order, key


groups, rows = {}, []
def gid(key, order):
    if key not in groups:
        groups[key] = dict(id=f"G{len(groups) + 1}", knob_order=order)
    return groups[key]["id"]

env = robosuite.make("KettleBoiling", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
RD = f"{ROOT}/runs/20261014_asus_seen5"
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "KettleBoiling" and "error" not in r}
for s in range(30):
    r = res[s]; z = np.load(r["scene"], allow_pickle=True); meta = json.loads(str(z["ep_meta"]))
    reset_to(env, str(z["xml"]), meta, z["state"]); env.unset_ep_meta()
    burners, order, key = geom(env); g = gid(key, order)
    set_state(env, np.load(f"{RD}/KettleBoiling/{r['tag']}_end.npz", allow_pickle=True)["state"])
    kp = np.array(env.sim.data.body_xpos[env.obj_body_id["obj"]])[:2]
    dist = {b: float(np.linalg.norm(p[:2] - kp)) for b, p in burners.items()}
    nb = min(dist, key=dist.get)
    on = [k for k in order if env.stove.is_burner_on(env=env, burner_loc=k)]
    rows.append(dict(scene=f"s{s}", ok=bool(r["success"]), layout=meta.get("layout_id"), group=g, nearest_burner=nb,
                     dist_cm=round(100 * dist[nb], 1), knobs_on=on, own_knob_idx=order.index(nb) if nb in order else None,
                     on_knob_idx=[order.index(k) for k in on]))
    print(json.dumps(rows[-1]), flush=True)
kettle_keys = set(groups)
sm = {}
SD = f"{ROOT}/runs/scenes_seen11_v2"
if all(os.path.exists(f"{SD}/SearingMeat_s{s}.npz") for s in range(30)):
    env.close()
    env = robosuite.make("SearingMeat", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
    for s in range(30):
        z = np.load(f"{SD}/SearingMeat_s{s}.npz", allow_pickle=True); meta = json.loads(str(z["ep_meta"]))
        reset_to(env, str(z["xml"]), meta, z["state"]); env.unset_ep_meta()
        _, order, key = geom(env); g = gid(key, order)
        sm.setdefault(g, []).append((f"s{s}", meta.get("layout_id")))
out = dict(rows=rows, groups={v["id"]: dict(knob_order=v["knob_order"], in_kettle=(k in kettle_keys)) for k, v in groups.items()},
           searingmeat_groups={g: v for g, v in sm.items()})
print(json.dumps(out["groups"])); print(json.dumps(out["searingmeat_groups"]))
json.dump(out, open(sys.argv[1], "w"), indent=1)
