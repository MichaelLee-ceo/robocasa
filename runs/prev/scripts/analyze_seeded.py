"""Seeded (fresh scenes) comparison: full instruction vs atomic chain, paired by seed; plus consistency checks."""
import json, glob, sys, os, collections
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomic_steps import TASKS

RUNS = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs")
SD, A, RS = f"{RUNS}/20261012_atomic_seeded", f"{RUNS}/20261012_atomic_decomp_seen", f"{RUNS}/20261012_groot_repro_seen_h1x"


def load(d, pat="results_w*.jsonl"):
    out = {}
    for p in glob.glob(f"{d}/{pat}"):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r:
                out[r["tag"]] = r
    return out


S = load(SD); D = load(A); REP = load(RS)
res = dict(tasks={})
tot = collections.Counter()
for t, steps in TASKS.items():
    b = {r["seed"]: r for r in S.values() if r["task"] == t and r["condition"] == "baseline_seed"}
    c = {r["seed"]: r for r in S.values() if r["task"] == t and r["condition"] == "chain_seed"}
    rep = {r["seed"]: r["success"] for r in REP.values() if r["task"] == t}
    seeds = sorted(set(b) & set(c))
    bo = sum(b[s]["success"] and not c[s]["success"] for s in seeds); co = sum(c[s]["success"] and not b[s]["success"] for s in seeds)
    T = dict(n=len(seeds), baseline=sum(b[s]["success"] for s in seeds), chain=sum(c[s]["success"] for s in seeds),
             baseline_only=bo, chain_only=co, repro_same_seeds=sum(rep[s] for s in seeds if s in rep),
             failure_point=dict(baseline=collections.Counter(), chain=collections.Counter()), steps=[])
    for name, RR in (("baseline", b), ("chain", c)):
        for s in seeds:
            r = RR[s]
            if not r["success"]:
                k = min(r["reached"], len(steps) - 1)
                T["failure_point"][name][f"step {k+1} {steps[k]['skill']}"] += 1
    for k, st in enumerate(steps):
        row = dict(k=k, skill=st["skill"])
        for name, RR in (("baseline", b), ("chain", c)):
            reached = [s for s in seeds if RR[s]["reached"] >= k or RR[s]["success"]]
            passed = [s for s in reached if RR[s]["reached"] > k or RR[s]["success"]]
            row[name] = [len(passed), len(reached)]
        T["steps"].append(row)
    T["failure_point"] = {k: dict(v) for k, v in T["failure_point"].items()}
    # demo-start control: chain code path with the task instruction at every step
    ct = [r for r in D.values() if r["task"] == t and r["condition"] == "chain_task"]
    bl = [r for r in D.values() if r["task"] == t and r["condition"] == "baseline"]
    T["demo_chain_task"] = [sum(r["success"] for r in ct), len(ct)]
    T["demo_baseline"] = [sum(r["success"] for r in bl), len(bl)]
    for kk in ("n", "baseline", "chain", "baseline_only", "chain_only", "repro_same_seeds"):
        tot[kk] += T[kk]
    tot["ct_k"] += T["demo_chain_task"][0]; tot["ct_n"] += T["demo_chain_task"][1]
    tot["bl_k"] += T["demo_baseline"][0]; tot["bl_n"] += T["demo_baseline"][1]
    res["tasks"][t] = T
tot = dict(tot)
d = tot["baseline_only"] + tot["chain_only"]
tot["mcnemar_p"] = round(binomtest(tot["chain_only"], d, 0.5).pvalue, 4) if d else None
res["total"] = tot
json.dump(res, open(f"{SD}/summary.json", "w"), indent=1)
for t, T in res["tasks"].items():
    print(f"{t:18s} seeded n={T['n']:2d}  full-instr {T['baseline']:2d}  chain {T['chain']:2d}  (repro same seeds {T['repro_same_seeds']})"
          f"  | demo: baseline {T['demo_baseline']}  chain_task-control {T['demo_chain_task']}")
    print("     steps:", [(s['skill'], f"base {s['baseline'][0]}/{s['baseline'][1]}", f"chain {s['chain'][0]}/{s['chain'][1]}") for s in T["steps"]])
    print("     failure point:", T["failure_point"])
print("TOTAL", tot)
