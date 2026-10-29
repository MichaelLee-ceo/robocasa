"""analyze_phase.py's tables, unchanged estimator, on the rows in use (effect_by_event.points(): RUNS -> FIX -> FIX3) instead
of one glob, so the README's per-task tables can be compared. Prints only. usage: python -I analyze_phase_current.py"""
import json, sys

R = "/home/michaellee/mclee/robocasa/runs"
sys.path.insert(0, f"{R}/20261028_critical")
import numpy as np  # noqa: E402
from effect_by_event import RUNS, points  # noqa: E402

rng = np.random.default_rng(0)


def summary(g):
    v = np.array([p["cb"] for p in g])
    lo, hi = np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)], [2.5, 97.5])
    mixed = sum(0 < p["y"].sum() < p["y"].size for p in g)
    gain = lambda x: 142 * np.sqrt(max(x, 0))
    return (f"n={len(g):3d} P(done)={np.mean([p['pd'] for p in g]):.2f} mixed={mixed:3d}  var_b {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
            f"best-of-8 gain {gain(v.mean()):.0f} pts [{gain(lo):.0f}, {gain(hi):.0f}]")


def main():
    hist = {}
    for line in open(f"{R}/20261028_critical/events.jsonl"):
        r = json.loads(line); hist[r["tag"]] = [t for t, _ in r["history"]]
    pts = points()
    for p in pts:
        h, k, t = hist[p["tag"]], p["k"], p["t"]
        start = 0 if k == 0 else h[k - 1]
        end = h[k] if k < len(h) else None
        p["el"], p["phase"] = t - start, ((t - start) / (end - start) if end else None)
    for task in RUNS:
        g0 = [p for p in pts if p["task"] == task]
        print(f"=== {task}\nall:", summary(g0))
        for k in range(4):
            for a, b, lab in [(0, .34, "early"), (.34, .67, "mid"), (.67, 1.01, "late")]:
                g = [p for p in g0 if p["k"] == k and p["phase"] is not None and a <= p["phase"] < b]
                if len(g) >= 8:
                    print(f"S{k} {lab:5s}", summary(g))
        for k in range(4):
            for a, b in [(0, 80), (80, 160), (160, 240), (240, 400), (400, 10 ** 6)]:
                g = [p for p in g0 if p["k"] == k and a <= p["el"] < b]
                if len(g) >= 8:
                    print(f"S{k} {a:4d}-{b if b < 10 ** 6 else '':4}", summary(g))


if __name__ == "__main__":
    main()
