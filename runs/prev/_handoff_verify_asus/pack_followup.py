"""PackIdenticalLunches follow-up (read-only, CPU only): what loose foods rest on, start base to tupperware, counters, food categories."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state
RD = f"{ROOT}/runs/20261014_asus_seen11"
env = robosuite.make("PackIdenticalLunches", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "PackIdenticalLunches" and "error" not in r}


def subtree(m, b):
    out = set()
    for i in range(m.nbody):
        j = i
        while j not in (0, b): j = m.body_parentid[j]
        if j == b: out.add(i)
    return out


def contacts(name):
    m, d = env.sim.model, env.sim.data; S = subtree(m, env.obj_body_id[name]); o = set()
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in S and b2 not in S: o.add(m.body_id2name(m.body_rootid[b2]))
        elif b2 in S and b1 not in S: o.add(m.body_id2name(m.body_rootid[b1]))
    return sorted(x for x in o if x)


Q = {0: ["vegetable1"], 1: ["meat1"], 3: ["vegetable0"], 4: ["meat0"], 9: ["vegetable0"], 11: ["meat0"], 14: ["meat0"], 15: ["meat0"],
     17: ["vegetable0"], 23: ["meat1"], 27: ["vegetable0", "vegetable1"], 29: ["vegetable0"]}
rows = []
for s in range(30):
    r = res[s]; z = np.load(r["scene"], allow_pickle=True); meta = json.loads(str(z["ep_meta"]))
    reset_to(env, str(z["xml"]), meta, z["state"]); env.unset_ep_meta()
    m, d = env.sim.model, env.sim.data; b0 = np.array(d.body_xpos[m.body_name2id("mobilebase0_base")])
    tp = {t: np.array(d.body_xpos[env.obj_body_id[t]]) for t in ("tupperware0", "tupperware1")}
    near = min(tp, key=lambda t: np.linalg.norm(b0[:2] - tp[t][:2]))
    on = {t: [n for n in contacts(t) if not n.startswith("tupperware")] for t in tp}
    cats = {}
    for o in ("vegetable0", "vegetable1", "meat0", "meat1"):
        c = next(c for c in meta["object_cfgs"] if c["name"] == o)
        cats[o] = c.get("info", {}).get("mjcf_path", "").split("/objects/")[-1]
    row = dict(scene=f"s{s}", ok=bool(r["success"]), start_base_to_nearer_tupperware_cm=round(100 * float(np.linalg.norm(b0[:2] - tp[near][:2])), 1),
               tupperwares_on=on, foods=cats)
    if s in Q:
        set_state(env, np.load(f"{RD}/PackIdenticalLunches/{r['tag']}_end.npz", allow_pickle=True)["state"])
        tp = {t: np.array(d.body_xpos[env.obj_body_id[t]]) for t in tp}
        row["loose"] = {}
        for o in Q[s]:
            p = np.array(d.body_xpos[env.obj_body_id[o]]); nt = min(tp, key=lambda t: np.linalg.norm(p[:2] - tp[t][:2]))
            row["loose"][o] = dict(z=round(float(p[2]), 2), touches=contacts(o), nearest=nt, xy_cm=round(100 * float(np.linalg.norm(p[:2] - tp[nt][:2])), 1))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
