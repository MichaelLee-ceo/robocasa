"""Estimator checks on the coarse points in use (effect_by_event.points(): RUNS -> FIX -> FIX3). Read-only.
1. The M = 2 covariance estimator: unbiased for Var(p_i)? (simulation)
2. The M = 4 bias correction p(1 - p) / (M - 1): unbiased? (simulation)
3. 142 = 100 x E[max of 8 standard normals]? (numerical)
4. 'best-of-8 gain' = 142 sqrt(mean var_b) pooled over points vs averaged over the README's own bins (Jensen: sqrt of a
   pooled mean > mean of sqrt when var_b is concentrated in some points).
5. Bootstrap unit: decision points vs episodes (the points of one main-run episode share its scene and trajectory).
usage: python -I check_estimators.py"""
import os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa/runs"
sys.path.insert(0, f"{R}/20261028_critical")
from effect_by_event import RUNS, points  # noqa: E402

rng = np.random.default_rng(1)


def cov_est(y):
    n = len(y); d0, d1 = y[:, 0] - y[:, 0].mean(), y[:, 1] - y[:, 1].mean()
    return (d0 * d1).sum() / (n - 1)


def var4(y):
    p = y.mean(1); M = y.shape[1]
    return p.var(ddof=1) - np.mean(p * (1 - p)) / (M - 1)


def sim_unbiased():
    print("1-2. unbiasedness (200k simulated points, 8 candidates, p_i ~ Beta)")
    for a, b in [(0.5, 0.5), (2, 2), (0.3, 3), (5, 1)]:
        p = rng.beta(a, b, size=(200000, 8))
        true = a * b / ((a + b) ** 2 * (a + b + 1))
        y2 = (rng.random((200000, 8, 2)) < p[..., None]).astype(float)
        y4 = (rng.random((200000, 8, 4)) < p[..., None]).astype(float)
        c2 = np.mean([cov_est(y) for y in y2[:50000]])
        v4 = np.mean([var4(y) for y in y4[:50000]])
        print(f"   Beta({a}, {b}): Var(p) {true:.4f}  M=2 covariance {c2:.4f}  M=4 corrected variance {v4:.4f}")


def e_max8():
    x = rng.standard_normal((2_000_000, 8))
    print(f"3. E[max of 8 N(0,1)] = {x.max(1).mean():.4f} (formula uses 1.42)")


def gain(v):
    return 142 * np.sqrt(max(v, 0))


def bins_readme(p, main_hist):
    """the README's analyze_phase bins: stage k x steps since the stage started"""
    k, t = p["k"], p["t"]
    hist = main_hist[p["tag"]]
    start = 0 if k == 0 else hist[k - 1][0]
    el = t - start
    for a, b in [(0, 80), (80, 160), (160, 240), (240, 400), (400, 10 ** 6)]:
        if a <= el < b:
            return (p["task"], k, a)


def jensen(pts, main_hist):
    print("4. pooled formula vs averages over finer groups (each group's gain from its own mean var_b, weighted by its size)")
    for task in list(RUNS) + ["all"]:
        g = [p for p in pts if task == "all" or p["task"] == task]
        v = np.array([p["cb"] for p in g])
        pooled = gain(v.mean())
        by = {}
        for p in g:
            by.setdefault(bins_readme(p, main_hist), []).append(p["cb"])
        binw = sum(len(x) * gain(np.mean(x)) for x in by.values()) / len(g)
        by_ep = {}
        for p in g:
            by_ep.setdefault(p["tag"], []).append(p["cb"])
        epw = sum(len(x) * gain(np.mean(x)) for x in by_ep.values()) / len(g)
        # one-rollout critic, cross-validated: pick the candidates whose continuation a finished, score on continuation b
        cv = []
        for p in g:
            y = p["y"]
            for a, b in ((0, 1), (1, 0)):
                pick = np.flatnonzero(y[:, a] == y[:, a].max())
                cv.append(y[pick, b].mean() - y[:, b].mean())
        print(f"   {task:21s} n {len(g):3d}  pooled 142 sqrt(mean var_b) {pooled:5.1f} pts | mean over stage x elapsed bins "
              f"({len(by)} bins) {binw:5.1f} | mean over episodes ({len(by_ep)}) {epw:5.1f} | one-rollout critic, cross-validated {100 * np.mean(cv):+5.1f}")


def bootstrap_unit(pts):
    print("5. 95 % CI of mean var_b: bootstrap over points (as reported) vs over episodes (clusters)")
    for task in list(RUNS) + ["all"]:
        g = [p for p in pts if task == "all" or p["task"] == task]
        v = np.array([p["cb"] for p in g])
        lo, hi = np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)], [2.5, 97.5])
        eps = {}
        for p in g:
            eps.setdefault(p["tag"], []).append(p["cb"])
        E = list(eps.values())
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(E), len(E))
            x = np.concatenate([E[i] for i in idx])
            bs.append(x.mean())
        elo, ehi = np.percentile(bs, [2.5, 97.5])
        print(f"   {task:21s} mean {v.mean():+.4f}  points [{lo:+.4f}, {hi:+.4f}] -> gain [{gain(lo):.0f}, {gain(hi):.0f}] | "
              f"episodes [{elo:+.4f}, {ehi:+.4f}] -> gain [{gain(elo):.0f}, {gain(ehi):.0f}]")


def main():
    import json
    main_hist = {}
    for line in open(f"{R}/20261028_critical/events.jsonl"):
        r = json.loads(line); main_hist[r["tag"]] = r["history"]
    sim_unbiased()
    e_max8()
    pts = points()
    jensen(pts, main_hist)
    bootstrap_unit(pts)


if __name__ == "__main__":
    main()
