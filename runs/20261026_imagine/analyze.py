"""Compare, per KettleBoiling scene: plain policy (4 seeds: policy_seed 0 from runs/20261025_seen480 + 1..3 here),
episode-level best-of-4 (any seed succeeds), and the stage-level imagine-then-choose search (one search per scene).
Cost is counted in policy calls (one per 16-step chunk, imagined or executed) and simulated steps.
usage: python analyze.py [out dir=full] [task=KettleBoiling]"""
import glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "full")
TASK = sys.argv[2] if len(sys.argv) > 2 else "KettleBoiling"
rs = [json.loads(l) for p in glob.glob(f"{OUT}/results_w*.jsonl") for l in open(p)]
errs = [r for r in rs if "error" in r]
rs = [r for r in rs if "error" not in r and r["tag"].startswith(TASK + "_")]
base = defaultdict(dict)
for p in glob.glob(f"{HERE}/../20261025_seen480/run/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if r.get("task") == TASK:
            base[r["tag"]][0] = dict(success=r["success"], steps=r["steps"])
for r in rs:
    if r["condition"] == "baseline":
        base[r["tag"]][r["policy_seed"]] = dict(success=r["success"], steps=r["steps"])
search = {r["tag"]: r for r in rs if r["condition"] == "imagine"}
seed_of = lambda t: int(t.split("_s")[1].split("_")[0])
tags = sorted(set(base) | set(search), key=seed_of)
print(f"{'scene':6s} {'base(4 seeds)':14s} {'search':7s} {'steps':>5s} {'cands':>5s} {'calls':>5s} backtracked  path")
tot = dict(p=0.0, n=0, any4=0, search=0, sn=0, calls_s=0, calls_b=0, greedy=0)
for t in tags:
    b = base.get(t, {})
    bs = [b[ps]["success"] for ps in sorted(b)]
    sr = search.get(t)
    if bs:
        tot["p"] += sum(bs) / len(bs); tot["n"] += 1; tot["any4"] += any(bs)
        tot["calls_b"] += sum(-(-b[ps]["steps"] // 16) for ps in b)
    if sr:
        tot["search"] += sr["success"]; tot["sn"] += 1; tot["calls_s"] += sr["policy_calls"]
        tot["greedy"] += sr["success"] and not sr["backtracked"]
    print(f"s{seed_of(t):<5d} {''.join('1' if x else '.' for x in bs):4s} ({sum(bs)}/{len(bs)})     "
          + (f"{'OK' if sr['success'] else 'fail':7s} {sr['steps']:5d} {sr['n_candidates']:5d} {sr['policy_calls']:5d} {str(sr['backtracked']):11s} {sr['path']}" if sr else "-"))
n = max(tot["n"], 1)
print(f"\nscenes with baseline: {tot['n']}, with search: {tot['sn']}, errors: {len(errs)}")
print(f"plain policy success (mean over 4 seeds per scene): {100 * tot['p'] / n:.1f}%")
print(f"episode-level best-of-4 (any seed succeeds):        {100 * tot['any4'] / n:.1f}%   policy calls {tot['calls_b']}")
print(f"stage-level search (N=8, keep 3):                   {100 * tot['search'] / max(tot['sn'], 1):.1f}%   policy calls {tot['calls_s']}")
print(f"  of which without backtracking (greedy path):      {100 * tot['greedy'] / max(tot['sn'], 1):.1f}%")
for e in errs[:3]:
    print("ERROR", e["tag"], e["condition"], e["error"])
