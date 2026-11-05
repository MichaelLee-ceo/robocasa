import json, sys, collections
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
err = [r for r in rows if "error" in r]
print("errors", err[:5])
rows = [r for r in rows if "error" not in r]
VAL = {"KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"}
by = collections.defaultdict(list)
for r in rows: by[r["task"]].append(r)
def stats(rs):
    t = np.array([r["true"] for r in rs]); g = np.array([r["rule"] for r in rs])
    e = np.abs(t - g) / 0.2
    sat = np.abs(t) > 0.999
    ws = (np.sign(g[sat]) != np.sign(t[sat])).mean() if sat.any() else np.nan
    big = (e >= 3).mean()
    return len(rs), e.mean(), ws, big, sat.mean()
print(f"{'task':22s} {'n':>5s} {'meanErr':>7s} {'wrongSat%':>9s} {'err>=3%':>7s}  branch mix (n, mean err, wrongSat%)")
for task in sorted(by, key=lambda x: (x not in VAL, x)):
    rs = by[task]
    n, me, ws, big, _ = stats(rs)
    bm = collections.defaultdict(list)
    for r in rs: bm[r["branch"]].append(r)
    s = "  ".join(f"{b}:{len(v)}/{stats(v)[1]:.2f}/{100*stats(v)[2]:.0f}" for b, v in sorted(bm.items()))
    print(f"{('*' if task in VAL else ' ')+task:22s} {n:5d} {me:7.2f} {100*ws:9.1f} {100*big:7.1f}  {s}")
for grp, name in ((VAL, "validated 3"), (set(by) - VAL, "other 13")):
    rs = [r for r in rows if r["task"] in grp]
    n, me, ws, big, _ = stats(rs)
    bm = collections.defaultdict(list)
    for r in rs: bm[r["branch"]].append(r)
    print(name, n, f"mean {me:.3f} wrongSat {100*ws:.2f}% err>=3 {100*big:.2f}%")
    for b, v in sorted(bm.items()):
        n, me, ws, big, _ = stats(v)
        print(f"   {b:18s} n={n:5d} frac={n/len(rs):.3f} meanErr={me:.2f} wrongSat={100*ws:.1f}% err>=3={100*big:.1f}%")
