"""Offline analysis of saved end states (no rollouts): which conjunct of the official _check_success is False at
the end of each failed episode whose last attempted step is the task's last step; plus PreSoakPan object poses
at handoff (single-horizon saved step states) vs at the end of the atomic chains."""
import os, sys, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Desktop/robocasa/third_party/Isaac-GR00T"))
import gymnasium as gym
import robocasa  # noqa
import groot_eval  # noqa
import robocasa.utils.object_utils as OU
from atomic_eval import load_scene
from oracle import set_state

OUT = os.path.expanduser("~/Desktop/robocasa/runs/20261013_scenes_v2")
NST = dict(KettleBoiling=2, PreSoakPan=3, StackBowlsCabinet=2, SteamInMicrowave=4, PrepareCoffee=2)
R = {}
for p in glob.glob(f"{OUT}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            R[r["tag"]] = r


def conj(t, e):
    if t == "PreSoakPan":
        return dict(water_on=e.sink.get_handle_state(env=e)["water_on"],
                    pan_in_sink=OU.obj_inside_of(e, "obj1", e.sink, partial_check=False),
                    sponge_in_sink=OU.obj_inside_of(e, "obj2", e.sink, partial_check=False),
                    far_pan=OU.gripper_obj_far(e, "obj1"), far_sponge=OU.gripper_obj_far(e, "obj2"))
    if t == "StackBowlsCabinet":
        return dict(b1_in_cab=OU.obj_inside_of(e, "bowl1", e.cabinet), b2_in_cab=OU.obj_inside_of(e, "bowl2", e.cabinet),
                    stacked=OU.check_obj_in_receptacle(e, "bowl2", "bowl1") or OU.check_obj_in_receptacle(e, "bowl1", "bowl2"),
                    far1=OU.gripper_obj_far(e, obj_name="bowl1"), far2=OU.gripper_obj_far(e, obj_name="bowl2"))
    if t == "KettleBoiling":
        kp = np.array(e.sim.data.body_xpos[e.obj_body_id[e.objects["obj"].name]])[:2]
        knobs = e.stove.get_knobs_state(env=e); best = (None, 9)
        for loc, site in e.stove.burner_sites.items():
            if site is not None:
                d = float(np.linalg.norm(np.array(e.sim.data.get_site_xpos(site.get("name")))[:2] - kp))
                if d < best[1]:
                    best = (loc, d)
        return dict(on_stove=OU.check_obj_fixture_contact(e, "obj", e.stove), on_burner_site=best[1] < 0.15,
                    burner_under_on=bool(best[0] in knobs and e.stove.is_burner_on(env=e, burner_loc=best[0])),
                    any_burner_on=any(e.stove.is_burner_on(env=e, burner_loc=l) for l in knobs), gripper_far=OU.gripper_obj_far(e))
    if t == "SteamInMicrowave":
        return dict(veg_in_bowl=OU.check_obj_in_receptacle(e, "vegetable", "bowl"), bowl_in_mw=OU.obj_inside_of(e, "bowl", e.microwave),
                    door_closed=e.microwave.is_closed(env=e), button_pressed="n/a (python flag, not in sim state)")
    if t == "PrepareCoffee":
        return dict(mug_placed=e.coffee_machine.check_receptacle_placement_for_pouring(e, "obj"), gripper_obj_far=OU.gripper_obj_far(e),
                    gripper_button_far=e.coffee_machine.gripper_button_far(e), turned_on="n/a (python flag, not in sim state)")


def pose(e, name):
    b = e.obj_body_id[name]
    return np.array(e.sim.data.body_xpos[b]), np.array(e.sim.data.body_xquat[b])


def grip_dist(e, name):
    g = e.sim.data.site_xpos[e.robots[0].eef_site_id["right"]]
    return float(np.linalg.norm(g - pose(e, name)[0]))


envs = {}
rows = []
pre = collections.defaultdict(list)
for tag, r in sorted(R.items()):
    t = r["task"]; n = NST[t]
    f = f"{OUT}/{t}/{tag}_end.npz"
    if not os.path.exists(f):
        continue
    cond = r["condition"]
    pre_need = t == "PreSoakPan" and cond.startswith("chain_seed")
    last_need = (not r["success"]) and ((cond.startswith("chain_seed") and r["reached"] == n - 1) or (cond.startswith("reset_seed_atomic") and r["step"] == n - 1))
    if not (pre_need or last_need):
        continue
    is_chain = cond.startswith("chain_seed"); is_solo = cond.startswith("reset_seed_atomic") or cond == "baseline_seed_save"
    model = "aseen" if cond.endswith("_ackpt") else ("pre" if cond.endswith("_pckpt") else "single")
    if t not in envs:
        for x in envs.values():
            x[0].close()
        envs = {t: (lambda b: (b, b.env))(gym.make(f"robocasa/{t}", split="target", enable_render=False).unwrapped)}
    b, e = envs[t]
    load_scene(e, r["scene"]); set_state(e, np.load(f, allow_pickle=True)["state"])
    last_failed = (not r["success"]) and ((is_chain and r["reached"] == n - 1) or (cond.startswith("reset_seed_atomic") and r["step"] == n - 1))
    if last_failed:
        rows.append(dict(task=t, model=model, kind="chain" if is_chain else "solo", seed=r["seed"],
                         **{k: (bool(v) if not isinstance(v, str) else v) for k, v in conj(t, e).items()}))
    if t == "PreSoakPan" and is_chain:
        pp, pq = pose(e, "obj1"); sp, _ = pose(e, "obj2")
        pre[model].append(dict(seed=r["seed"], reached=r["reached"], success=r["success"],
                               pan_in_sink=bool(OU.obj_inside_of(e, "obj1", e.sink, partial_check=False)),
                               sponge_in_sink=bool(OU.obj_inside_of(e, "obj2", e.sink, partial_check=False)),
                               pan_tilt_deg=round(float(np.degrees(2 * np.arccos(min(1, abs(pq[0]))))), 1),
                               grip_pan=round(grip_dist(e, "obj1"), 3), grip_sponge=round(grip_dist(e, "obj2"), 3)))

# single-horizon handoff states for PreSoakPan (saved when steps 1 and 2 completed)
hand = []
b, e = (lambda b: (b, b.env))(gym.make("robocasa/PreSoakPan", split="target", enable_render=False).unwrapped)
for s in range(20):
    tag = f"PreSoakPan_s{s}_single"
    f = f"{OUT}/PreSoakPan/{tag}_states.npz"
    if not os.path.exists(f):
        continue
    for rec in np.load(f, allow_pickle=True)["recs"]:
        if int(rec["k"]) in (1, 2):
            load_scene(e, R[tag]["scene"]); set_state(e, rec["state"])
            _, pq = pose(e, "obj1")
            hand.append(dict(seed=s, k=int(rec["k"]), t=int(rec["t"]), pan_in_sink=bool(OU.obj_inside_of(e, "obj1", e.sink, partial_check=False)),
                             sponge_in_sink=bool(OU.obj_inside_of(e, "obj2", e.sink, partial_check=False)),
                             pan_tilt_deg=round(float(np.degrees(2 * np.arccos(min(1, abs(pq[0]))))), 1),
                             grip_pan=round(grip_dist(e, "obj1"), 3), grip_sponge=round(grip_dist(e, "obj2"), 3)))
json.dump(dict(last_step_failures=rows, presoak_chain_end=pre, presoak_single_handoff=hand),
          open(f"{OUT}/end_state_analysis.json", "w"), indent=1, default=str)

print("== last-step failures: count of False per condition")
for (t, m, kind), g in sorted(collections.defaultdict(list, {}).items()):
    pass
G = collections.defaultdict(list)
for x in rows:
    G[(x["task"], x["model"], x["kind"])].append(x)
for k, g in sorted(G.items()):
    keys = [c for c in g[0] if c not in ("task", "model", "kind", "seed")]
    print(k, "n=%d" % len(g), {c: sum(1 for x in g if x[c] is False) for c in keys})
print("== PreSoak chain end states")
for m, g in pre.items():
    for x in sorted(g, key=lambda x: x["seed"]):
        print(" ", m, x)
print("== PreSoak single handoff states")
for x in hand:
    print(" ", x)
