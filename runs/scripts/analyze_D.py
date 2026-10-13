"""Experiment D: on fresh seeded scenes, restore the exact state where the continuous full-instruction run
(baseline_seed_save) finished step k-1, then run step k alone with (a) the atomic instruction and (b) the full
instruction. (b) vs the continuous run's own step-k outcome is the restore sanity check."""
import json, glob, os, sys
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomic_steps import TASKS

SD = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/20261012_atomic_seeded")
R = {}
for p in glob.glob(f"{SD}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            R[r["tag"]] = r


def mcnemar(a, b):
    x = sum(a[s] and not b[s] for s in a); y = sum(b[s] and not a[s] for s in a)
    return dict(a_only=x, b_only=y, p=round(binomtest(x, x + y, .5).pvalue, 4) if x + y else None)


out = {}
for t, ks in (("StackBowlsCabinet", [1]), ("SteamInMicrowave", [1, 2, 3])):
    steps = TASKS[t]
    full = {r["seed"]: r for r in R.values() if r["task"] == t and r["condition"] == "baseline_seed_save"}
    T = dict(full=dict(success=sum(r["success"] for r in full.values()), n=len(full)), steps={})
    for k in ks:
        ra = {r["seed"]: r["step_success"] for r in R.values() if r["task"] == t and r["condition"] == "reset_seed_atomic" and r["step"] == k}
        rt = {r["seed"]: r["step_success"] for r in R.values() if r["task"] == t and r["condition"] == "reset_seed_task" and r["step"] == k}
        seeds = sorted(set(ra) & set(rt))
        cont = {s: bool(full[s]["reached"] > k or full[s]["success"]) for s in seeds}
        ra = {s: ra[s] for s in seeds}; rt = {s: rt[s] for s in seeds}
        T["steps"][k] = dict(skill=steps[k]["skill"], n=len(seeds), continuous=sum(cont.values()),
                             restore_task=sum(rt.values()), restore_atomic=sum(ra.values()),
                             sanity_task_vs_continuous=mcnemar(rt, cont),
                             atomic_vs_task=mcnemar(ra, rt))
    out[t] = T
json.dump(out, open(f"{SD}/summary_D.json", "w"), ensure_ascii=False, indent=1)
for t, T in out.items():
    print("==", t, "full (continuous, saved runs):", f"{T['full']['success']}/{T['full']['n']}")
    for k, v in T["steps"].items():
        print(f"  step {k+1} {v['skill']:18s} n={v['n']:2d}  continuous {v['continuous']:2d}  restore+full {v['restore_task']:2d}  "
              f"restore+atomic {v['restore_atomic']:2d}   sanity {v['sanity_task_vs_continuous']}  atomic_vs_full {v['atomic_vs_task']}")
