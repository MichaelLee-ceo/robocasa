"""Per-episode records for the composite-seen stage page (v2, replaces export_episodes.py).
done[k]  = env step at which tracker step k first passed (runs/scripts/atomic_steps.py, sequential, official checks)
kitchen  = "layout <id> · <sink model>" (sink model only for tasks that use the sink)
note     = raw end-state facts, written for failed episodes, and for successful ones only when the official and the
           corrected containment test disagree. Containment is reported with robocasa's OU.obj_inside_of ("official":
           un-normalised edge vectors, so the 0.05 tolerance is 0.05 / edge length in metres) and, if different, with
           a corrected test (unit edge vectors, 0.05 m), e.g. "sponge in sink (official) / not in sink (0.05 m)".
PreSoakPan also gets sponge_start_in_sink: official test on the scene's initial state.
usage: python -I export_episodes2.py <out.json> <task[,task...]|all> <run_dir> [<run_dir> ...]   (env needs EGL on GPU0)"""
import glob, json, os, re, sys
import numpy as np
import gymnasium as gym

ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import robocasa.utils.object_utils as OU
import robosuite.utils.transform_utils as T
import groot_eval  # noqa  (registers robocasa/* envs)
from oracle import reset_to, set_state
from atomic_steps import TASKS

SINK_TASKS = {"PreSoakPan", "WashLettuce", "RinseSinkBasin"}
# (object, fixture attribute, label) pairs whose containment is reported in the note
CONTAIN = {
    "PreSoakPan": [("obj1", "sink", "pan", "sink"), ("obj2", "sink", "sponge", "sink")],
    "WashLettuce": [("lettuce", "sink", "lettuce", "sink")],
    "StackBowlsCabinet": [("bowl1", "cabinet", "bowl1", "cabinet"), ("bowl2", "cabinet", "bowl2", "cabinet")],
    "SteamInMicrowave": [("bowl", "microwave", "bowl", "microwave")],
}


def inside_fixed(env, obj_name, fixture, th=0.05):
    """OU.obj_inside_of with unit edge vectors, so th is in metres (same 8 bbox corners, any interior region)."""
    obj = env.objects[obj_name]
    pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
    quat = T.convert_quat(env.sim.data.body_xquat[env.obj_body_id[obj.name]], to="xyzw")
    pts = obj.get_bbox_points(trans=pos, rot=quat)
    for p0, px, py, pz in fixture.get_int_sites(relative=False).values():
        ok = True
        for q in (px, py, pz):
            e = q - p0; L = np.linalg.norm(e); eh = e / L
            if not all(-th <= np.dot(p - p0, eh) <= L + th for p in pts):
                ok = False; break
        if ok:
            return True
    return False


def contain_phrase(env, obj, fx, label, flabel):
    a = bool(OU.obj_inside_of(env, obj, getattr(env, fx), partial_check=False))
    b = inside_fixed(env, obj, getattr(env, fx))
    word = lambda x: f"in {flabel}" if x else f"not in {flabel}"
    return (f"{label} {word(a)}" if a == b else f"{label} {word(a)} (official) / {word(b)} (0.05 m)"), a != b


def gripper_dist(env, obj):
    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    return float(np.linalg.norm(eef - env.sim.data.body_xpos[env.obj_body_id[obj]]))


def end_facts(task, env, S, steps):
    parts, differ = [], False
    for c in CONTAIN.get(task, []):
        try:
            s, d = contain_phrase(env, *c); parts.append(s); differ |= d
        except Exception as e:  # object or fixture missing in this task variant
            print(f"  containment {c} failed: {e!r}", file=sys.stderr)
    if task in SINK_TASKS:
        hs = env.sink.get_handle_state(env=env)
        q = float(env.sim.data.qpos[env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(f"{env.sink.name}_handle_joint")]])
        parts.append(f"faucet q={q:.3f} water {'on' if hs['water_on'] else 'off'}")
    if task == "WashLettuce":
        parts.append(f"lettuce under water {'yes' if env.sink.check_obj_under_water(env, 'lettuce') else 'no'}")
        parts.append(f"gripper {gripper_dist(env, 'lettuce'):.2f} m from lettuce")
    if task == "PreSoakPan":
        parts.append(f"gripper {gripper_dist(env, 'obj1'):.2f} m from pan, {gripper_dist(env, 'obj2'):.2f} m from sponge")
    if task not in CONTAIN and task not in SINK_TASKS:  # no hand-written facts: which tracker checks hold at the end
        for k, st in enumerate(steps[:-1]):
            try:
                parts.append(f"{st['skill']} check {'holds' if st['check'](env, dict(S)) else 'fails'}")
            except Exception as e:
                print(f"  check {k} failed: {e!r}", file=sys.stderr)
    return ", ".join(parts), differ


out, which, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
recs = []
for rd in runs:
    for p in sorted(glob.glob(f"{rd}/results_w*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r and r["task"] in TASKS and (which == "all" or r["task"] in which.split(",")):
                recs.append((rd, r))
recs.sort(key=lambda x: (x[1]["task"], x[1]["seed"]))
envs, rows = {}, []
for rd, r in recs:
    task, steps = r["task"], TASKS[r["task"]]
    z = np.load(r["scene"], allow_pickle=True)
    meta, xml = json.loads(str(z["ep_meta"])), str(z["xml"])
    sinks = sorted(set(re.findall(r'file="[^"]*/fixtures/sinks/([^/"]+)/', xml)))
    kitchen = f"layout {meta.get('layout_id')}" + (f" · {';'.join(sinks)}" if task in SINK_TASKS and sinks else "")
    row = dict(task=task, scene=f"s{r['seed']}", ok=bool(r["success"]), end=int(r["steps"]), horizon=int(r["horizon"]),
               done=[dict((k, t) for t, k in r["history"]).get(k) for k in range(len(steps))],
               note="", kitchen=kitchen)
    if row["ok"] and row["done"][-1] is None:  # official success (reward > 0 at some step inside a 16-step chunk) that no
        row["done"][-1] = row["end"]            # longer held at the chunk end where the tracker checks: last step = end
    try:
        if task not in envs:
            for e in envs.values():
                e.close()
            envs = {task: gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped}
        kenv = envs[task].env
        reset_to(kenv, xml, meta, z["state"]); kenv.unset_ep_meta()
        if task == "PreSoakPan":
            row["sponge_start_in_sink"] = bool(OU.obj_inside_of(kenv, "obj2", kenv.sink, partial_check=False))
        end = np.load(f"{rd}/{task}/{r['tag']}_end.npz", allow_pickle=True)
        set_state(kenv, end["state"])
        S = dict(end["S"].item()) if end["S"].shape == () else {}
        note, differ = end_facts(task, kenv, S, steps)
        if not row["ok"] or differ:
            row["note"] = note
    except Exception as e:  # leave the note empty rather than guess
        print(f"note failed for {r['tag']}: {e!r}", file=sys.stderr)
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
print(f"{len(rows)} records -> {out}")
