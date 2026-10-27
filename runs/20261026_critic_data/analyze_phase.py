"""Does the candidate chunk change the stage outcome, and where in the stage? (clean data: hidden="settled" runs)

For each decision point, the two continuations of a candidate share its first chunk and differ only afterwards, so
the covariance of the two continuations' outcomes across candidates estimates the between-candidate variance of
P(stage done | candidate) without bias (zero if the candidate does not matter). Reported per position in the stage, with
95 % bootstrap intervals over decision points, plus the same/cross agreement and the gain of a perfect best-of-8 picker
(about 1.42 SD of the candidates' success probabilities for 8 draws).
usage: python analyze_phase.py "<glob of results_w*.jsonl>" [main run dir]"""
import glob, json, sys
import numpy as np

rng = np.random.default_rng(0)
files = sorted(glob.glob(sys.argv[1]))
MAIN = sys.argv[2] if len(sys.argv) > 2 else "../20261025_seen480/run"
main = {r["tag"]: r for p in glob.glob(f"{MAIN}/results_w*.jsonl") for r in map(json.loads, open(p))}
rows = {}
for p in files:
    for r in map(json.loads, open(p)):
        if "error" not in r:
            rows[(r["tag"], r["t"])] = r
pts = []
for r in rows.values():
    hist = [t for t, _ in main[r["tag"]]["history"]]
    k, t = r["k"], r["t"]
    start = 0 if k == 0 else hist[k - 1]
    end = hist[k] if k < len(hist) else None
    y = np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float)
    n = len(y); d0, d1 = y[:, 0] - y[:, 0].mean(), y[:, 1] - y[:, 1].mean()
    pts.append(dict(k=k, el=t - start, phase=(t - start) / (end - start) if end else None, y=y,
                    cb=(d0 * d1).sum() / (n - 1), pd=y.mean()))
print(f"{len(pts)} decision points from {len(files)} files")


def summary(g):
    v = np.array([p["cb"] for p in g])
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    Y = [p["y"] for p in g]
    same = np.mean([y[i, 0] == y[i, 1] for y in Y for i in range(len(y))])
    cross = np.mean([y[i, a] == y[j, b] for y in Y for i in range(len(y)) for j in range(len(y)) if i != j for a in (0, 1) for b in (0, 1)])
    mixed = sum(0 < y.sum() < y.size for y in Y)
    gain = lambda x: 142 * np.sqrt(max(x, 0))
    return (f"n={len(g):3d} P(done)={np.mean([p['pd'] for p in g]):.2f} mixed={mixed:3d}  same {same:.2f} vs cross {cross:.2f}  "
            f"var_b {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  best-of-8 gain {gain(v.mean()):.0f} pts [{gain(lo):.0f}, {gain(hi):.0f}]")


print("all:", summary(pts))
print("--- by position in the stage (main run finished that stage; fraction of the stage elapsed)")
for k in range(4):
    for a, b, lab in [(0, .34, "early"), (.34, .67, "mid"), (.67, 1.01, "late")]:
        g = [p for p in pts if p["k"] == k and p["phase"] is not None and a <= p["phase"] < b]
        if len(g) >= 8:
            print(f"S{k} {lab:5s}", summary(g))
print("--- by steps since the stage started (all points)")
for k in range(4):
    for a, b in [(0, 80), (80, 160), (160, 240), (240, 400), (400, 10 ** 6)]:
        g = [p for p in pts if p["k"] == k and a <= p["el"] < b]
        if len(g) >= 8:
            print(f"S{k} {a:4d}-{b if b < 10 ** 6 else '':4}", summary(g))
