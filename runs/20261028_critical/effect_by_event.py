"""Is the candidate chunk's effect larger right before a physical event? Uses the uniform decision points already
collected (the coarse rows in use, data.py) and the events of events.py.

Per decision point, the between-candidate variance of P(stage done), var_b, is estimated without bias by the
covariance of the two continuations across candidates (as ../20261026_critic_data/analyze_phase.py). Points are grouped
by the time from the point to the next event of a kind in the same stage. The candidate chunk applies actions t..t+15
and produces the states t+1..t+16, so the events it can cause have their step f in (t, t + 16]; an event at f = t was
caused by the chunk before.
Effect per group (summary()), with 95 % bootstrap CIs over points:
  one-rollout critic, cross-validated (cv_gain; the headline): pick the candidates whose continuation 0 finished, score
           them on continuation 1, and the reverse. Unbiased and 0 where the chunk has no effect, but it is the gain of a
           weak critic that sees one outcome per candidate.
  perfect best-of-8, per point: a perfect picker among 8 candidates, computed per point from its own mean and var_b
           (best_of_8) and averaged. With 2 continuations var_b is noisy, so this reads about +3 pts even where the
           chunk has no effect (simulation in _verify_20261029); differences of a few points mean nothing.
  pooled: 142 sqrt(mean var_b), as reported before. It is the gain at a point of average variance and overstates the
           mean gain, since var_b sits in a minority of points (Jensen).
usage: python -I effect_by_event.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
from data import DATA, HERE, OUT, TASKS  # noqa: E402,F401

KINDS = {"grasp (touch, hold, close)": {"touch", "hold", "close"}, "fixture (touch, move)": {"fx_touch", "fx_move"},
         "place (at, release, open)": {"at", "release", "open"}, "any of these": {"touch", "hold", "close", "fx_touch", "fx_move", "at", "release", "open"}}
BINS = [(0, 16, "in the chunk"), (16, 48, "1-2 chunks later"), (48, 128, "3-7 chunks later"), (128, 10 ** 6, "8+ chunks later"), (None, None, "none")]  # (a, b]
rng = np.random.default_rng(0)


def best_of_8(q, v, n=8):
    """gain of a perfect best-of-n picker at points with mean P(done) q and between-candidate variance v (arrays): the
    candidates' P(done) ~ Beta with that mean and variance (v clipped to [0, q(1 - q)]), E[max of n] - q. Unlike
    1.42 sqrt(v) it respects 0 <= P(done) <= 1."""
    from scipy.stats import beta
    q, v = np.atleast_1d(np.asarray(q, float)), np.atleast_1d(np.asarray(v, float))
    out = np.zeros(len(q))
    m = (q > 0) & (q < 1) & (v > 0)
    if m.any():
        s = q[m] * (1 - q[m]) / np.minimum(v[m], 0.999 * q[m] * (1 - q[m])) - 1
        x = (np.arange(2000) + 0.5) / 2000  # E[max] = integral of 1 - F(x)^n over [0, 1]
        F = beta.cdf(x[None], (q[m] * s)[:, None], ((1 - q[m]) * s)[:, None])
        out[m] = np.maximum((1 - F ** n).mean(1) - q[m], 0)
    return out


def cv_gain(y):
    """cross-validated gain of a one-rollout critic at a point (candidates x 2): pick the candidates whose continuation a
    finished (all of them if none did), score them on continuation b, minus the point's rate on b; both ways, averaged"""
    g = []
    for a, b in ((0, 1), (1, 0)):
        pick = y[:, a] == y[:, a].max()
        g.append(y[pick, b].mean() - y[:, b].mean())
    return float(np.mean(g))


def points():
    """the coarse decision points in use, with var_b (cb), the per-point gains and the main run's events in the stage"""
    ev = {r["tag"]: r for r in map(json.loads, open(os.path.join(HERE, "events.jsonl")))}
    out = []
    for task in TASKS:
        for (tag, t), r in sorted(data.rows(task).items()):
            y = data.outcomes(r)
            n = len(y); d0, d1 = y[:, 0] - y[:, 0].mean(), y[:, 1] - y[:, 1].mean()
            out.append(dict(task=task, tag=tag, t=t, k=r["k"], y=y, cb=(d0 * d1).sum() / (n - 1), pd=y.mean(), cv=cv_gain(y),
                            events=[e for e in ev[tag]["events"] if e["k"] == r["k"]], roll=data.roll(r), snap=data.snap(r),
                            dir=r["_dir"], py_state=r["py_state"], hidden=r.get("hidden")))
    for p, g in zip(out, best_of_8([p["pd"] for p in out], [p["cb"] for p in out])):
        p["gain"] = float(g)
    return out


def summary(g):
    if len(g) < 8:
        return f"n={len(g):3d}"
    v, G, C = (np.array([p[k] for p in g]) for k in ("cb", "gain", "cv"))
    idx = [rng.integers(0, len(v), len(v)) for _ in range(2000)]
    ci = lambda x: np.percentile([x[i].mean() for i in idx], [2.5, 97.5])
    (lo, hi), (glo, ghi), (clo, chi) = ci(v), 100 * ci(G), 100 * ci(C)
    mixed = sum(0 < p["y"].sum() < p["y"].size for p in g)
    return (f"n={len(g):3d} P(done)={np.mean([p['pd'] for p in g]):.2f} mixed={mixed:3d}  one-rollout critic (cv) {100 * C.mean():+4.1f} pts "
            f"[{clo:+.1f}, {chi:+.1f}]  perfect best-of-8 per point {100 * G.mean():4.1f} [{glo:.1f}, {ghi:.1f}]  "
            f"var_b {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (pooled 142 sqrt: {142 * np.sqrt(max(v.mean(), 0)):2.0f})")


def next_dt(p, types):
    """steps from the point to the next event of these types that its chunk or a later one can cause (f > t)"""
    dts = [e["t"] - p["t"] for e in p["events"] if e["type"] in types and e["t"] > p["t"]]
    return min(dts) if dts else None


def main():
    pts = points()
    print(f"{len(pts)} decision points: " + ", ".join(f"{t} {sum(p['task'] == t for p in pts)}" for t in TASKS))
    for kind, types in KINDS.items():
        print(f"--- next {kind} event in the same stage, all tasks")
        for a, b, lab in BINS:
            g = [p for p in pts if (next_dt(p, types) is None if a is None else (next_dt(p, types) is not None and a < next_dt(p, types) <= b))]
            print(f"  {lab:17s}", summary(g))
    print("--- next grasp event within 48 steps (this chunk or the next two), per task")
    for task in TASKS:
        for lab, cond in [("within 48", lambda d: d is not None and d <= 48), ("later/none", lambda d: d is None or d > 48)]:
            g = [p for p in pts if p["task"] == task and cond(next_dt(p, KINDS["grasp (touch, hold, close)"]))]
            print(f"  {task:21s} {lab:10s}", summary(g))


if __name__ == "__main__":
    main()
