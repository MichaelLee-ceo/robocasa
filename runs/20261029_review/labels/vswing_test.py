"""Task 1, second pass: a significance test for local swings, since the strict thresholds of vseries.py flag nothing.

For every interior point b of a series (neighbours a before, c after; also windows of up to 3 points on each side):
  dip   : b lower than the highest point among the w before AND lower than the highest among the w after
  spike : b higher than the lowest among the w before AND the lowest among the w after
Intersection-union test: p = max(one-sided Fisher exact p of b vs the chosen point before, b vs the chosen point after),
times w * w for choosing the extreme points (Bonferroni over the choices). A swing is flagged at p < ALPHA.
Expected number of flags from binomial noise alone: parametric bootstrap with a monotone truth per series (weighted
isotonic fit, increasing or decreasing, whichever fits better), 1000 replicates, the same test.
Writes swing_test.json. usage: python vswing_test.py"""
import json, os, sys
import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, TASKS3  # noqa: E402
from vseries import series, monotone_truth  # noqa: E402

ALPHA = 0.01
W = 3
_cache = {}


def fisher_less(x1, n1, x2, n2):
    """one-sided p that rate 1 < rate 2"""
    key = (x1, n1, x2, n2)
    if key not in _cache:
        _cache[key] = fisher_exact([[x1, n1 - x1], [x2, n2 - x2]], alternative="less")[1]
    return _cache[key]


def swings(xs, ns, ts=None):
    """xs, ns: counts along one series. Returns flagged swings (dicts)."""
    out = []
    n = len(xs)
    V = [x / m for x, m in zip(xs, ns)]
    for b in range(1, n - 1):
        best = None
        for w in range(1, W + 1):
            lo_a, hi_c = max(0, b - w), min(n, b + w + 1)
            before, after = list(range(lo_a, b)), list(range(b + 1, hi_c))
            if len(before) < w and len(after) < w and w > 1:
                continue
            for kind in ("dip", "spike"):
                if kind == "dip":
                    a = max(before, key=lambda i: V[i]); c = max(after, key=lambda i: V[i])
                    if not (V[a] > V[b] < V[c]):
                        continue
                    p = max(fisher_less(xs[b], ns[b], xs[a], ns[a]), fisher_less(xs[b], ns[b], xs[c], ns[c]))
                else:
                    a = min(before, key=lambda i: V[i]); c = min(after, key=lambda i: V[i])
                    if not (V[a] < V[b] > V[c]):
                        continue
                    p = max(fisher_less(xs[a], ns[a], xs[b], ns[b]), fisher_less(xs[c], ns[c], xs[b], ns[b]))
                p = min(1.0, p * len(before) * len(after))
                if best is None or p < best["p"]:
                    best = dict(b=b, a=a, c=c, kind=kind, p=p, w=w)
        if best is not None and best["p"] < ALPHA:
            out.append(best)
    return out


def main():
    pts = json.load(open(os.path.join(HERE, "vseries.json")))["points"]
    by = series(pts)
    rng = np.random.default_rng(1)
    res = {"flags": [], "summary": {}}
    for task in TASKS3:
        keys = [k for k in by if k[0] == task and len(by[k]) >= 3]
        obs = []
        for key in keys:
            s = by[key]
            for f in swings([p["x"] for p in s], [p["n"] for p in s]):
                f.update(task=task, tag=key[1], k=key[2],
                         pts=[dict(t=s[i]["t"], x=s[i]["x"], n=s[i]["n"], V=round(s[i]["V"], 3), dir=s[i]["dir"], hidden=s[i]["hidden"])
                              for i in (f["a"], f["b"], f["c"])],
                         series=[(s[i]["t"], round(s[i]["V"], 2), s[i]["n"]) for i in range(max(0, f["b"] - 4), min(len(s), f["b"] + 5))])
                obs.append(f)
        truths = {key: monotone_truth(by[key]) for key in keys}
        boot = []
        for _ in range(1000):
            c = 0
            for key in keys:
                ns = [p["n"] for p in by[key]]
                xs = rng.binomial(ns, truths[key])
                c += len(swings(list(xs), ns))
            boot.append(c)
        boot = np.array(boot)
        res["summary"][task] = dict(series=len(keys), interior_points=sum(len(by[k]) - 2 for k in keys), flagged=len(obs),
                                    flagged_dip=sum(f["kind"] == "dip" for f in obs), flagged_spike=sum(f["kind"] == "spike" for f in obs),
                                    null_mean=float(boot.mean()), null_p95=float(np.percentile(boot, 95)),
                                    p_null_ge_obs=float((boot >= len(obs)).mean()))
        res["flags"] += obs
        print(task, res["summary"][task])
        for f in sorted(obs, key=lambda f: f["p"]):
            print(f"   {f['kind']:5s} {f['tag']:30s} S{f['k']} p={f['p']:.1e} " + " ".join(f"{q['t']}:{q['x']}/{q['n']}" for q in f["pts"])
                  + "   | " + " ".join(f"{t}:{v}" for t, v, _ in f["series"]))
    json.dump(res, open(os.path.join(HERE, "swing_test.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
