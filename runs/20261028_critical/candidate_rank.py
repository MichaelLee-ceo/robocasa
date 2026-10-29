"""Does a candidate's first chunk, compared with its 7 siblings at the same decision point, predict its own outcome?
(split_analysis.py found losers that swing the arm 40-90 cm in one chunk while carrying, and drop the object.)

Per candidate, from its first chunk (16 steps) and relative to the 8 candidates of its point:
  mag     length of the summed end_effector_position command / the median over the 8
  peak    largest single-step end_effector_position command / the median over the 8 of the same
  rot     length of the summed end_effector_rotation command / the median over the 8
  atyp    distance of the summed end_effector_position command from the 8's median / the median such distance
  grip    the candidate's mean gripper_close command differs from the majority of the 8 by more than 0.5
Outcome: the candidate's success (mean of its 2 continuations) minus the point's mean; points where every rollout
ended the same way are left out (nothing to rank). The chunks come from the rollouts of the coarse row in use (data.py).
usage: python -I candidate_rank.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TASKS  # noqa: E402
from effect_by_event import points  # noqa: E402

rng = np.random.default_rng(0)


def features(chunks):
    P = np.stack([c["action.end_effector_position"][0] for c in chunks])  # 8 x 16 x 3
    Rr = np.stack([c["action.end_effector_rotation"][0] for c in chunks])
    G = np.stack([c["action.gripper_close"][0, :, 0] for c in chunks])
    S = P.sum(1)
    rel = lambda v: v / (np.median(v) + 1e-6)
    med = np.median(S, 0)
    d = np.linalg.norm(S - med, axis=1)
    g = G.mean(1)
    return dict(mag=rel(np.linalg.norm(S, axis=1)), peak=rel(np.linalg.norm(P, axis=2).max(1)), rot=rel(np.linalg.norm(Rr.sum(1), axis=1)),
                atyp=rel(d), grip=(np.abs(g - np.median(g)) > 0.5).astype(float))


def main():
    rows = []
    for p in points():
        y = p["y"].mean(1)
        if y.min() == y.max() and p["y"].min() == p["y"].max():
            continue
        f = features([np.load(f"{p['roll']}/{p['tag']}_t{p['t']}_c{i}_m0.npz") for i in range(len(y))])
        for i in range(len(y)):
            rows.append(dict(task=p["task"], k=p["k"], s=float(y[i]), rel=float(y[i] - y.mean()), **{k: float(v[i]) for k, v in f.items()}))
    print(f"{len(rows)} candidates at {len(rows) // 8} points with mixed outcomes")
    bins = {"mag": [0, 0.5, 0.8, 1.25, 2, 3, np.inf], "peak": [0, 0.8, 1.25, 2, 3, np.inf], "rot": [0, 0.5, 0.8, 1.25, 2, 3, np.inf],
            "atyp": [0, 0.5, 1, 2, 3, 5, np.inf], "grip": [-0.5, 0.5, 1.5]}
    for f, edges in bins.items():
        print(f"--- {f}: candidate success minus its point's mean (percentage points), 95 % bootstrap CI")
        for a, b in zip(edges[:-1], edges[1:]):
            g = np.array([r["rel"] for r in rows if a <= r[f] < b])
            if len(g) < 10:
                print(f"  [{a}, {b}) n {len(g)}"); continue
            bs = [g[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sub = {t: np.mean([r["rel"] for r in rows if a <= r[f] < b and r["task"] == t]) * 100 for t in TASKS}
            print(f"  [{a}, {b}) n {len(g):4d}  {100 * g.mean():+5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]   "
                  + "  ".join(f"{t[:5]} {v:+.0f}" for t, v in sub.items() if not np.isnan(v)))


if __name__ == "__main__":
    main()
