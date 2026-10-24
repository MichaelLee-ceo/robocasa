"""Stove burner centres and knob positions (stove frame and world), and kettle displacement. Read-only, CPU only."""
import json, os, re, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to, set_state
RD = f"{ROOT}/runs/20261014_asus_seen5"
env = robosuite.make("KettleBoiling", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w") for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "KettleBoiling" and "error" not in r}
out = {}
for s in (5, 12, 28):
    r = res[s]; z = np.load(r["scene"], allow_pickle=True); xml = str(z["xml"])
    reset_to(env, xml, json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    st = env.stove; m, d = env.sim.model, env.sim.data
    bid = m.body_name2id(st.root_body); Rm = d.body_xmat[bid].reshape(3, 3); c = d.body_xpos[bid]
    loc = lambda p: [round(100 * float(x), 1) for x in Rm.T @ (np.array(p) - c)]
    burners = {k: np.array(d.get_site_xpos(v.get("name"))) for k, v in st.burner_sites.items() if v is not None}
    knobs = {}
    for k, j in st.knob_joints.items():
        if j is None: continue
        jid = m.joint_name2id(f"{st.naming_prefix}knob_{k}_joint")
        knobs[k] = np.array(d.xanchor[jid])
    near = {b: min(knobs, key=lambda k: np.linalg.norm(knobs[k][:2] - bp[:2])) for b, bp in burners.items()}
    out[f"s{s}"] = dict(stove_model=";".join(sorted(set(re.findall(r'file="[^"]*/fixtures/stoves/([^/"]+)/', xml)))) or None,
                        layout=json.loads(str(z["ep_meta"])).get("layout_id"),
                        burners_stove_frame_cm={k: loc(v) for k, v in burners.items()},
                        knobs_stove_frame_cm={k: loc(v) for k, v in knobs.items()},
                        knob_order_along_stove_x=sorted(knobs, key=lambda k: (Rm.T @ (knobs[k] - c))[0]),
                        nearest_knob_to_burner=near)
    print(json.dumps({f"s{s}": out[f"s{s}"]}), flush=True)
r = res[26]; z = np.load(r["scene"], allow_pickle=True)
reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
p0 = np.array(env.sim.data.body_xpos[env.obj_body_id["obj"]])
set_state(env, np.load(f"{RD}/KettleBoiling/{r['tag']}_end.npz", allow_pickle=True)["state"])
p1 = np.array(env.sim.data.body_xpos[env.obj_body_id["obj"]])
out["s26_kettle"] = dict(start_world_m=[round(float(x), 3) for x in p0], end_world_m=[round(float(x), 3) for x in p1],
                         displacement_cm=round(100 * float(np.linalg.norm(p1 - p0)), 1), dxyz_cm=[round(100 * float(x), 1) for x in p1 - p0])
print(json.dumps({"s26_kettle": out["s26_kettle"]}))
json.dump(out, open(sys.argv[1], "w"), indent=1)
