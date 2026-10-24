"""Raw end-state notes for PrepareCoffee (all episodes) and KettleBoiling (failed episodes), from each episode's end npz
reloaded on its frozen scene. Coffee: dispenser placement check alone, mug-to-dispenser horizontal/vertical distance,
mug tilt, gripper distance/contact/grasp. The coffee machine's turned_on is a Python-level latch (set the first time the
gripper touches the start button) and is not in the saved MuJoCo state, so it is not reported. Kettle: kettle-stove
contact, nearest burner and horizontal distance, every knob angle and on/off, gripper distance/contact/grasp.
usage: python -I raw_coffee_kettle.py <out.json> <run_dir>"""
import glob, json, os, sys
import numpy as np
import gymnasium as gym

ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T
import groot_eval  # noqa
from oracle import reset_to, set_state
from atomic_steps import TASKS


def grip(env, obj):
    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    d = float(np.linalg.norm(eef - env.sim.data.body_xpos[env.obj_body_id[obj]]))
    touch = bool(env.check_contact(env.robots[0].gripper["right"], env.objects[obj]))
    try:
        held = bool(OU.check_obj_grasped(env, obj))
    except Exception:
        held = None
    return f"gripper {d:.2f} m from {obj}, touching {'yes' if touch else 'no'}, grasped {'yes' if held else 'no' if held is not None else 'n/a'}"


def tilt_deg(env, obj):
    q = T.convert_quat(env.sim.data.body_xquat[env.obj_body_id[obj]], to="xyzw")
    z = T.quat2mat(q)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))


def coffee(env):
    cm = env.coffee_machine
    sid = env.sim.model.site_name2id(f"{cm.naming_prefix}receptacle_place_site")
    sp, op = env.sim.data.site_xpos[sid], env.sim.data.body_xpos[env.obj_body_id["obj"]]
    place = bool(cm.check_receptacle_placement_for_pouring(env, "obj"))
    return (f"mug under dispenser {'yes' if place else 'no'} (xy {100 * np.linalg.norm(op[:2] - sp[:2]):.1f} cm, need <4; "
            f"dz {100 * abs(op[2] - sp[2]):.1f} cm, need <10), mug tilt {tilt_deg(env, 'obj'):.0f} deg, {grip(env, 'obj')}, "
            f"gripper far from button {'yes' if cm.gripper_button_far(env) else 'no'} (>0.15 m)")


def kettle(env):
    st = env.stove
    kp = np.array(env.sim.data.body_xpos[env.obj_body_id["obj"]])[:2]
    on_stove = bool(OU.check_obj_fixture_contact(env, "obj", st))
    dists = {loc: float(np.linalg.norm(np.array(env.sim.data.get_site_xpos(s.get("name")))[:2] - kp))
             for loc, s in st.burner_sites.items() if s is not None}
    near = min(dists, key=dists.get)
    knobs = st.get_knobs_state(env=env)
    ks = ", ".join(f"{loc} {q:.2f} rad {'ON' if st.is_burner_on(env=env, burner_loc=loc) else 'off'}" for loc, q in sorted(knobs.items()))
    return (f"kettle on stove {'yes' if on_stove else 'no'}, nearest burner {near} {100 * dists[near]:.1f} cm (need <15), "
            f"knobs: {ks}, {grip(env, 'obj')}")


out, rd = sys.argv[1], sys.argv[2]
rows, envs = [], {}
recs = sorted((json.loads(l) for p in glob.glob(f"{rd}/results_w*.jsonl") for l in open(p)), key=lambda r: (r["task"], r["seed"]))
for r in recs:
    if "error" in r or r["task"] not in ("PrepareCoffee", "KettleBoiling"):
        continue
    if r["task"] == "KettleBoiling" and r["success"]:
        continue
    task = r["task"]
    done = [dict((k, t) for t, k in r["history"]).get(k) for k in range(len(TASKS[task]))]
    if r["success"] and done[-1] is None:
        done[-1] = r["steps"]
    try:
        if task not in envs:
            for e in envs.values():
                e.close()
            envs = {task: gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped}
        env = envs[task].env
        z = np.load(r["scene"], allow_pickle=True)
        reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); env.unset_ep_meta()
        set_state(env, np.load(f"{rd}/{task}/{r['tag']}_end.npz", allow_pickle=True)["state"])
        note = coffee(env) if task == "PrepareCoffee" else kettle(env)
    except Exception as e:
        note = ""; print(f"failed {r['tag']}: {e!r}", file=sys.stderr)
    row = dict(task=task, scene=f"s{r['seed']}", ok=bool(r["success"]), done=done, note=note)
    rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
print(f"{len(rows)} rows -> {out}")
