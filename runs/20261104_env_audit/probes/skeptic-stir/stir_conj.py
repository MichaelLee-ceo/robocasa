"""Per-frame replay of StirVegetables seen480 failures: which conjunct of the official stir counter blocks.
usage: python -I stir_conj.py 14,22,28"""
import glob, json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
import robocasa.utils.object_utils as OU  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa

task = "StirVegetables"
seeds = [int(x) for x in sys.argv[1].split(",")]
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
cls_detect = type(env)._detect_stirring
LAST = {}
def wrapped(names, movement_threshold=0.05):
    r = cls_detect(env, names, movement_threshold)
    LAST["stirred"] = bool(r)
    return r
env._detect_stirring = wrapped

for seed in seeds:
    r = [x for x in rows if x["task"] == task and x["seed"] == seed and "error" not in x][0]
    z = np.load(local(r["scene"]), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    for v in ("veg1", "veg2"):
        if hasattr(env, f"prev_obj_pos_{v}"):
            delattr(env, f"prev_obj_pos_{v}")
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
    knob = env.knob
    site = env.stove.burner_sites[knob]
    T = len(states)
    fr = []
    for f in range(T):
        LAST.clear()
        set_state(env, states[f])
        grasped = bool(OU.check_obj_grasped(env, "spatula"))
        contact = bool(OU.check_obj_fixture_contact(env, "pot", env.stove))
        pos = np.array(env.sim.data.body_xpos[env.obj_body_id["pot"]])
        dist = float(np.linalg.norm(np.array(env.sim.data.get_site_xpos(site.get("name")))[:2] - pos[:2]))
        ka = abs(env.stove.get_knobs_state(env=env)[knob]); kon = bool(0.35 <= ka <= 2 * np.pi - 0.35)
        loc = env.stove.check_obj_location_on_stove(env, "pot", threshold=0.15)
        v1 = bool(OU.check_obj_in_receptacle(env, "veg1", "pot")); v2 = bool(OU.check_obj_in_receptacle(env, "veg2", "pot"))
        tilt = float(np.degrees(np.arccos(np.clip(np.array(env.sim.data.body_xmat[env.obj_body_id["pot"]]).reshape(3, 3)[2, 2], -1, 1))))
        fr.append(dict(f=f, g=grasped, c=contact, d=round(dist, 3), kon=kon, loc_ok=loc == knob, v1=v1, v2=v2,
                       st=LAST.get("stirred"), stime=int(env.success_time), z=round(float(pos[2]), 3), tilt=round(tilt, 1)))
    # first frame pot_loc becomes false after being true
    first_off = next((x["f"] for x in fr if not x["loc_ok"]), None)
    reasons = dict(no_contact=sum(not x["c"] for x in fr), far=sum(x["d"] >= 0.15 for x in fr), knob_off=sum(not x["kon"] for x in fr))
    g_frames = [x for x in fr if x["g"]]
    stirred_frames = [x for x in fr if x["st"]]
    blocked = [x["f"] for x in fr if x["g"] and x["st"] and not (x["loc_ok"] and x["v1"] and x["v2"])]
    blocked_by_pot = [x["f"] for x in fr if x["g"] and x["st"] and x["v1"] and x["v2"] and not x["loc_ok"]]
    print(json.dumps(dict(tag=r["tag"], live_success=r["success"], knob=knob, T=T, first_pot_off=first_off,
                          pot_off_frames=sum(not x["loc_ok"] for x in fr), reasons=reasons,
                          grasp_frames=len(g_frames), first_grasp=g_frames[0]["f"] if g_frames else None,
                          stirred_frames=len(stirred_frames), stirred_at=[x["f"] for x in stirred_frames][:40],
                          blocked=blocked[:40], n_blocked=len(blocked), n_blocked_by_pot=len(blocked_by_pot),
                          stime_end=fr[-1]["stime"],
                          around_off=[fr[i] for i in range(max(0, (first_off or 0) - 3), min(T, (first_off or 0) + 3))] if first_off else None,
                          end=fr[-1])), flush=True)
