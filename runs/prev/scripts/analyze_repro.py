import json, glob, math, csv, sys, collections
from scipy.stats import binomtest
R = sys.argv[1]
P = json.load(open(sorted(glob.glob(f"{R}/paper_table11_*.json"))[0]))
rows = []
for p in glob.glob(f"{R}/results_w*.jsonl") + glob.glob(f"{R}/retry/results_w*.jsonl"):
    rows += [json.loads(l) for l in open(p)]
ok = {}
for r in rows:
    if "error" not in r:
        ok[r["tag"]] = r
rows = sorted(ok.values(), key=lambda r: (r["task"], r["seed"]))
for r in rows:
    r["split"] = "composite_seen" if "seen_h1x" in R else "composite_unseen"; r["condition"] = "baseline_seeded_h1x"
json.dump(rows, open(f"{R}/results.json", "w"), indent=1)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
out = []
for t, rs in by.items():
    k = sum(r["success"] for r in rs); n = len(rs)
    lo, hi = wilson(k, n); pp = P[t] / 100
    pv = binomtest(k, n, pp).pvalue
    succ_steps = [r["first_success_step"] for r in rs if r["success"]]
    out.append(dict(task=t, n=n, success=k, rate=round(k / n, 3), ci95_lo=round(lo, 3), ci95_hi=round(hi, 3),
                    paper_rate=pp, binom_p_vs_paper=round(pv, 4), horizon=rs[0]["horizon"],
                    median_success_step=sorted(succ_steps)[len(succ_steps) // 2] if succ_steps else None,
                    mean_wall_s=round(sum(r["wall_s"] for r in rs) / n, 1)))
out.sort(key=lambda x: -x["paper_rate"])
with open(f"{R}/summary_by_task.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
K = sum(o["success"] for o in out); N = sum(o["n"] for o in out)
lo, hi = wilson(K, N)
macro = sum(o["rate"] for o in out) / len(out)
for o in out:
    flag = " *" if o["binom_p_vs_paper"] < 0.05 else ""
    print(f"{o['task']:22s} {o['success']:2d}/{o['n']}  {o['rate']:.0%} [{o['ci95_lo']:.0%},{o['ci95_hi']:.0%}]  paper {o['paper_rate']:.0%}  p={o['binom_p_vs_paper']:.3f}{flag}")
print(f"OVERALL micro {K}/{N} = {K/N:.1%} [{lo:.1%},{hi:.1%}]; macro avg over tasks = {macro:.1%}; paper {P['average']}%")
json.dump(dict(per_task=out, overall=dict(success=K, n=N, rate=K / N, ci95=[lo, hi], macro=macro, paper=P["average"]/100)),
          open(f"{R}/summary.json", "w"), indent=1)
