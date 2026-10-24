"""One JSON record per episode for the composite-seen stage page.
done[k] = env step at which tracker step k (runs/scripts/atomic_steps.py, checked sequentially in atomic_eval.rollout)
first passed, null if never. For failed episodes, note = which step checks hold at the final state (reloaded from the
episode's _end.npz on its frozen scene, each check evaluated on its own, not sequentially).
usage: python -I export_episodes.py <out.json> <run_dir> [<run_dir> ...]   (CPU only, no rendering)"""
import glob, json, os, sys
import numpy as np
import gymnasium as gym

ROOT = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{ROOT}/runs/scripts")
import robocasa  # noqa
import groot_eval  # noqa  (registers robocasa/* envs)
from oracle import reset_to, set_state
from atomic_steps import TASKS

ORDER = ["StackBowlsCabinet", "SteamInMicrowave", "KettleBoiling", "PreSoakPan", "PrepareCoffee"]
# what each step check means, phrased as an end-state fact (holds / does not hold)
FACT = {
    "StackBowlsCabinet": [("a bowl in cabinet", "no bowl in cabinet"), ("task complete", "task not complete")],
    "SteamInMicrowave": [("vegetable in bowl", "vegetable not in bowl"), ("bowl in microwave", "bowl not in microwave"),
                         ("door closed", "door open"), ("microwave on", "microwave not on")],
    "KettleBoiling": [("kettle on burner", "kettle not on burner"), ("task complete", "task not complete")],
    "PreSoakPan": [("pan in sink", "pan not in sink"), ("sponge in sink", "sponge not in sink"), ("task complete", "task not complete")],
    "PrepareCoffee": [("mug under dispenser", "mug not under dispenser"), ("machine started", "machine not started")],
}

NO_LAST = {"SteamInMicrowave", "PrepareCoffee"}


def raw_presoak(kenv):
    """PreSoakPan end state from raw quantities, not from the tracker's step checks (those also require the gripper
    to be >0.25 m away, and the last one is the whole success check). "in sink" is robocasa's OU.obj_inside_of
    (all 8 bbox corners, tolerance not normalised: see audit_presoak/REPORT.md)."""
    import robocasa.utils.object_utils as OU
    hs = kenv.sink.get_handle_state(env=kenv)
    eef = kenv.sim.data.site_xpos[kenv.robots[0].eef_site_id["right"]]
    d = {o: float(np.linalg.norm(eef - kenv.sim.data.body_xpos[kenv.obj_body_id[o]])) for o in ("obj1", "obj2")}
    return ", ".join([
        "pan in sink" if OU.obj_inside_of(kenv, "obj1", kenv.sink, partial_check=False) else "pan not in sink",
        "sponge in sink" if OU.obj_inside_of(kenv, "obj2", kenv.sink, partial_check=False) else "sponge not in sink",
        f"faucet {'on' if hs['water_on'] else 'off'}",
        f"gripper {min(d.values()):.2f} m from nearest of pan/sponge"])
out, runs = sys.argv[1], sys.argv[2:]
recs = []
for rd in runs:
    for p in sorted(glob.glob(f"{rd}/results_w*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r and r["task"] in TASKS:
                recs.append((rd, r))
recs.sort(key=lambda x: (ORDER.index(x[1]["task"]), x[1]["seed"]))
envs, rows = {}, []
for rd, r in recs:
    task, steps = r["task"], TASKS[r["task"]]
    done = {k: t for t, k in r["history"]}
    note = ""
    if not r["success"]:
        try:
            if task not in envs:
                for e in envs.values():
                    e.close()
                envs = {task: gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped}
            kenv = envs[task].env
            z = np.load(r["scene"], allow_pickle=True)
            reset_to(kenv, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"]); kenv.unset_ep_meta()
            end = np.load(f"{rd}/{task}/{r['tag']}_end.npz", allow_pickle=True)
            set_state(kenv, end["state"])
            S = dict(end["S"].item()) if end["S"].shape == () else {}
            # microwave / coffee machine "on" is Python-level fixture state, not in the saved MuJoCo state: skip those last steps
            ks = range(len(steps) - 1) if task in NO_LAST else range(len(steps))
            if task == "PreSoakPan":
                note = raw_presoak(kenv)
            elif task in FACT:
                holds = {k: bool(steps[k]["check"](kenv, dict(S))) for k in ks}
                note = ", ".join(FACT[task][k][0 if h else 1] for k, h in holds.items())
            else:  # other seen tasks: which tracker step checks hold at the end (last one = official success check)
                holds = {k: bool(steps[k]["check"](kenv, dict(S))) for k in range(len(steps))}
                note = ", ".join(f"{steps[k]['skill']} check {'holds' if h else 'fails'}" for k, h in holds.items())
        except Exception as e:  # leave the note empty rather than guess
            note = ""
            print(f"note failed for {r['tag']}: {e!r}", file=sys.stderr)
    rows.append(dict(task=task, scene=f"s{r['seed']}", ok=bool(r["success"]), end=int(r["steps"]), horizon=int(r["horizon"]),
                     done=[done.get(k) for k in range(len(steps))], note=note))
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
print(f"{len(rows)} records -> {out}")
