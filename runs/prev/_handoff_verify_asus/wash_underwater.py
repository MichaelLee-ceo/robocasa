"""WashLettuce end states vs sink.check_obj_under_water's own quantities (read-only, CPU only: env built with robosuite.make,
no renderer, no camera observations, so no EGL/GPU context). usage: python -I wash_underwater.py <out.json> <seed> ..."""
import json, os, sys
import numpy as np
import robosuite
ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
from robosuite.utils.mjcf_utils import string_to_array
from oracle import reset_to, set_state

RD = f"{ROOT}/runs/20261014_asus_seen11"
env = robosuite.make("WashLettuce", robots="PandaOmron", has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, ignore_done=True)
res = {r["seed"]: r for p in sorted(os.listdir(RD)) if p.startswith("results_w")
       for r in map(json.loads, open(f"{RD}/{p}")) if r["task"] == "WashLettuce" and "error" not in r}
rows = []
for s in map(int, sys.argv[2:]):
    r = res[s]
    z = np.load(r["scene"], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
    set_state(env, np.load(f"{RD}/WashLettuce/{r['tag']}_end.npz", allow_pickle=True)["state"])
    sink = env.sink
    lp = np.array(env.sim.data.body_xpos[env.obj_body_id["lettuce"]])
    wid = env.sim.model.site_name2id(sink.water_site.get("name"))
    wp = np.array(env.sim.data.site_xpos[wid]); wsize = string_to_array(sink.water_site.get("size"))
    rad = float(env.objects["lettuce"].horizontal_radius)
    # sink frame: rotate world xy by -yaw of the sink body (fixture x = width axis, y = depth axis)
    bid = env.sim.model.body_name2id(sink.root_body)
    Rm = env.sim.data.body_xmat[bid].reshape(3, 3)
    d_local = Rm.T @ (lp - wp)
    regions = sink.get_int_sites(relative=False)

    def which(pt):
        out = []
        for name, (p0, px, py, pz) in regions.items():
            ok = True
            for q in (px, py):
                e = q - p0; L = np.linalg.norm(e); eh = e / L
                t = np.dot(pt - p0, eh)
                ok &= (-1e-9 <= t <= L + 1e-9)
            if ok:
                out.append(name)
        return out or ["none"]
    grip = env.robots[0].gripper["right"]
    hold_col = bool(env.check_contact(grip, env.objects["lettuce_container"]))
    hold_let = bool(env.check_contact(grip, env.objects["lettuce"]))
    row = dict(scene=f"s{s}", ok=bool(r["success"]), sink_model=None,
               xy_dist_cm=round(100 * float(np.linalg.norm(lp[:2] - wp[:2])), 1), xy_thresh_cm=round(100 * rad, 1),
               z_margin_cm=round(100 * float(lp[2] - (wp[2] + wsize[1])), 1),
               water_on=bool(sink.get_handle_state(env=env)["water_on"]),
               under_water=bool(sink.check_obj_under_water(env, "lettuce")),
               lettuce_minus_stream_sink_frame_cm=[round(100 * float(x), 1) for x in d_local],
               regions=list(regions.keys()), stream_over=which(wp), lettuce_in=which(lp),
               gripper_touch_colander=hold_col, gripper_touch_lettuce=hold_let)
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
