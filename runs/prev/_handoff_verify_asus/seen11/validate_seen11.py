"""Demo-replay validation of atomic_steps_seen11.TASKS (approach of runs/scripts/validate_atomic_steps.py).
usage: python -I validate_seen11.py TASK [n_demos=10] [extra_ep ...]
Per demo: reset_to (fresh scene + fresh Python fixture state), then for every frame: set_state (runs update_state),
env._check_success() once (what env.step's reward() does), then the sequential tracker (step k checked only after k-1).
Separately (after its own reset_to) checks every step predicate is False at the start frame of its annotated segment.
Writes val_<TASK>.json next to this file."""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ["ROBOCASA_ROOT"] + "/runs/scripts"))
import robosuite, robocasa  # noqa
from robosuite.controllers import load_composite_controller_config
from demo_utils import load_demo, n_episodes
from oracle import reset_to, set_state
from atomic_steps_seen11 import TASKS, step_segments

task = sys.argv[1]
nd = int(sys.argv[2]) if len(sys.argv) > 2 else 10
extra = [int(x) for x in sys.argv[3:]]
k = robosuite.make(env_name=task, robots="PandaOmron",
                   controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
                   has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True,
                   use_object_obs=True, obj_instance_split="target",
                   layout_and_style_ids=list(zip(range(1, 11), range(1, 11))))
N = n_episodes(task)
eps = [int(i * N / nd) for i in range(nd)] + [e for e in extra if e not in [int(i * N / nd) for i in range(nd)]]
steps = TASKS[task]
out = []
for ep in eps:
    t0 = time.time()
    d = load_demo(task, ep)
    units = step_segments(d)
    rec = dict(ep=ep, length=d["length"], annotated_steps=[u["skill"] for u in units],
               step_skills=[s["skill"] for s in steps], extra=ep in extra)
    rec["annotation_matches"] = rec["annotated_steps"] == rec["step_skills"]
    # 1) every predicate False at the start of its annotated segment (fresh scene per probe: no Python state carried)
    fas = []
    for i, s in enumerate(steps):
        if i < len(units):
            reset_to(k, d["xml"], d["ep_meta"]); set_state(k, d["states"][units[i]["start"]])
            fas.append(not bool(s["check"](k, {})))
        else:
            fas.append(None)
    rec["false_at_seg_start"] = fas
    # 2) sequential replay
    reset_to(k, d["xml"], d["ep_meta"])
    S, cur, fired = {}, 0, []
    succ_first = None
    for f in range(len(d["states"])):
        set_state(k, d["states"][f])
        if k._check_success() and succ_first is None:
            succ_first = f
        while cur < len(steps) and steps[cur]["check"](k, S):
            fired.append(f); cur += 1
    rec.update(all_passed=cur == len(steps), n_passed=cur, first_pass_frame=fired, env_success_first_frame=succ_first,
               final_env_success=bool(k._check_success()),
               seg_start=[u["start"] for u in units], seg_end=[u["end"] for u in units],
               S={kk: (v if not isinstance(v, np.generic) else v.item()) for kk, v in S.items()})
    rec["pass_minus_seg_end"] = [fired[i] - units[i]["end"] for i in range(min(len(fired), len(units)))]
    if not rec["all_passed"]:
        rec["failure"] = f"step {cur} ({steps[cur]['skill']}) never passed"
    rec["instr"] = [s["instr"](k, S) for s in steps]
    rec["wall_s"] = round(time.time() - t0, 1)
    out.append(rec)
    print(json.dumps({x: rec[x] for x in ("ep", "all_passed", "first_pass_frame", "seg_start", "seg_end", "false_at_seg_start",
                                         "env_success_first_frame", "annotation_matches", "wall_s")}), flush=True)
json.dump(out, open(os.path.join(HERE, f"val_{task}.json"), "w"), indent=1, default=str)
print("SUMMARY", task, sum(r["all_passed"] for r in out if not r["extra"]), "/", sum(not r["extra"] for r in out),
      "extra:", [(r["ep"], r["all_passed"]) for r in out if r["extra"]])
