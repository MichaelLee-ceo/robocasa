import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]; sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from oracle import reset_to
rep = {"G1": 0, "G2": 1, "G3": 4, "G4": 5, "G5": 7, "G6": 11, "G7": 16}
env = robosuite.make("KettleBoiling", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True)
out = {}
for g, s in rep.items():
    p = f"{ROOT}/runs/scenes_seen5_v2/KettleBoiling_s{s}.npz"
    z = np.load(p, allow_pickle=True); reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"])
    st, m, d = env.stove, env.sim.model, env.sim.data
    bid = m.body_name2id(st.root_body); Rm = d.body_xmat[bid].reshape(3, 3); c = d.body_xpos[bid]
    X = lambda p: round(100 * float((Rm.T @ (np.array(p) - c))[0]), 1)
    burners = {k: X(d.get_site_xpos(v.get("name"))) for k, v in st.burner_sites.items() if v is not None}
    knobs = {k: X(d.xanchor[m.joint_name2id(f"{st.naming_prefix}knob_{k}_joint")]) for k, j in st.knob_joints.items() if j is not None}
    fl = "front_left" if "front_left" in burners else "left"
    order = sorted(knobs, key=knobs.get)
    out[g] = dict(rep_scene=f"s{s}", burner=fl, burner_x=burners[fl], burners_x=burners, knobs_x={k: knobs[k] for k in order},
                  dx_leftmost_two={k: round(knobs[k] - burners[fl], 1) for k in order[:2]})
    print(g, json.dumps(out[g]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
