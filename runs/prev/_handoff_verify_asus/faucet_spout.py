"""Faucet handle / spout state at the start (frozen scene, no steps) and at the end (end npz) of sink-task episodes, and
sink geometry (handle-joint body to spout head, horizontal). Read-only, CPU only (robosuite.make without any renderer).
usage: python -I faucet_spout.py <out.json> "<Task>:<run_dir>:<seed,seed,...>" ..."""
import json, os, re, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state


def wrap_deg(q):
    d = np.degrees(q) % 360.0
    return round(float(d - 360.0 if d > 180.0 else d), 1)


def state(env):
    s = env.sink
    hs = s.get_handle_state(env=env)
    return dict(handle_q=round(float(hs["handle_joint"]), 3), water_on=bool(hs["water_on"]),
                spout_deg=wrap_deg(hs["spout_joint"]), spout_ori=hs["spout_ori"])


def geometry(env):
    """horizontal distance from the handle joint's body to the spout head (the spout body's geom farthest from the
    spout joint axis), and spout reach (that geom's horizontal distance from the spout joint)."""
    m, d, p = env.sim.model, env.sim.data, env.sink.naming_prefix
    hj, sj = m.joint_name2id(f"{p}handle_joint"), m.joint_name2id(f"{p}spout_joint")
    hb, sb = m.jnt_bodyid[hj], m.jnt_bodyid[sj]
    hpos = d.xanchor[hj]; spos = d.xanchor[sj]
    # all geoms in the spout body subtree
    sub = [b for b in range(m.nbody) if b == sb or any(a == sb for a in _anc(m, b))]
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub]
    head = max(gs, key=lambda g: np.linalg.norm(d.geom_xpos[g][:2] - spos[:2])) if gs else None
    hsub = [b for b in range(m.nbody) if b == hb or any(a == hb for a in _anc(m, b))]
    hgs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in hsub]
    tip = max(hgs, key=lambda g: np.linalg.norm(d.geom_xpos[g][:2] - hpos[:2])) if hgs else None
    out = dict(handle_anchor_to_spout_anchor_cm=round(100 * float(np.linalg.norm(hpos[:2] - spos[:2])), 1))
    if head is not None:
        out["spout_reach_cm"] = round(100 * float(np.linalg.norm(d.geom_xpos[head][:2] - spos[:2])), 1)
    if head is not None and tip is not None:
        out["lever_tip_to_spout_head_cm"] = round(100 * float(np.linalg.norm(d.geom_xpos[tip][:2] - d.geom_xpos[head][:2])), 1)
        out["lever_length_cm"] = round(100 * float(np.linalg.norm(d.geom_xpos[tip][:2] - hpos[:2])), 1)
    return out


def _anc(m, b):
    while b != 0:
        b = m.body_parentid[b]
        yield b


rows, geo, envs = [], {}, {}
for spec in sys.argv[2:]:
    task, rd, seeds = spec.split(":")
    res = {r["seed"]: r for p in sorted(os.listdir(rd)) if p.startswith("results_w")
           for r in map(json.loads, open(f"{rd}/{p}")) if r["task"] == task and "error" not in r}
    if task not in envs:
        envs = {task: robosuite.make(task, robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False,
                                     use_camera_obs=False, ignore_done=True)}
    env = envs[task]
    for s in map(int, seeds.split(",")):
        r = res[s]
        z = np.load(r["scene"], allow_pickle=True)
        xml = str(z["xml"])
        sink = ";".join(sorted(set(re.findall(r'file="[^"]*/fixtures/sinks/([^/"]+)/', xml))))
        reset_to(env, xml, json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
        st0 = state(env)
        if sink not in geo:
            try:
                geo[sink] = geometry(env)
            except Exception as e:
                geo[sink] = dict(error=repr(e))
        set_state(env, np.load(f"{rd}/{task}/{r['tag']}_end.npz", allow_pickle=True)["state"])
        st1 = state(env)
        row = dict(task=task, scene=f"s{s}", ok=bool(r["success"]), sink=sink,
                   handle_q_end=st1["handle_q"], water_on_end=st1["water_on"],
                   spout_deg_start=st0["spout_deg"], spout_deg_end=st1["spout_deg"], spout_ori_end=st1["spout_ori"])
        rows.append(row); print(json.dumps(row), flush=True)
json.dump(dict(rows=rows, geometry=geo), open(sys.argv[1], "w"), indent=1)
print(json.dumps(geo))
