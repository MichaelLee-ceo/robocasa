"""Experiments A (atomic step 1, then full instruction) and B (atomic with missing info added), paired by seed
with the existing full-instruction and atomic-chain runs on fresh scenes."""
import json, glob, os, sys, collections
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomic_steps import TASKS

SD = os.path.expanduser("~/Desktop/robocasa/runs/20261012_atomic_seeded")
R = {}
for p in glob.glob(f"{SD}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            R[r["tag"]] = r
CONDS = [("baseline_seed", "整句"), ("chain_seed", "atomic 串接"), ("chain_then_task_seed", "A: 第1步 atomic → 整句"),
         ("chain_aug_seed", "B: atomic 補資訊")]
out = {}
for t in ("StackBowlsCabinet", "SteamInMicrowave"):
    steps = TASKS[t]; T = {}
    by = {c: {r["seed"]: r for r in R.values() if r["task"] == t and r["condition"] == c} for c, _ in CONDS}
    seeds = sorted(set.intersection(*[set(v) for v in by.values()]))
    for c, name in CONDS:
        rr = by[c]
        row = dict(name=name, success=sum(rr[s]["success"] for s in seeds), n=len(seeds), steps=[])
        for k, st in enumerate(steps):
            reached = [s for s in seeds if rr[s]["reached"] >= k or rr[s]["success"]]
            passed = [s for s in reached if rr[s]["reached"] > k or rr[s]["success"]]
            row["steps"].append([len(passed), len(reached)])
        T[c] = row
    tests = {}
    for a, b in (("chain_then_task_seed", "baseline_seed"), ("chain_then_task_seed", "chain_seed"),
                 ("chain_aug_seed", "baseline_seed"), ("chain_aug_seed", "chain_seed")):
        x = sum(by[a][s]["success"] and not by[b][s]["success"] for s in seeds)
        y = sum(by[b][s]["success"] and not by[a][s]["success"] for s in seeds)
        tests[f"{a} vs {b}"] = dict(a_only=x, b_only=y, p=round(binomtest(x, x + y, .5).pvalue, 4) if x + y else None)
    T["tests"] = tests; T["skills"] = [s["skill"] for s in steps]
    out[t] = T
json.dump(out, open(f"{SD}/summary_AB.json", "w"), ensure_ascii=False, indent=1)
for t, T in out.items():
    print("==", t, "steps:", T["skills"])
    for c, name in CONDS:
        row = T[c]
        print(f"  {name:22s} {row['success']:2d}/{row['n']}   steps (passed/reached): {['%d/%d' % tuple(x) for x in row['steps']]}")
    for k, v in T["tests"].items():
        print("   ", k, v)
