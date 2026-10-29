"""At the critical points, what separates the successful candidates (noises) from the failing ones?

The noise fixes the chunk (GR00T's flow-matching action head is deterministic given the observation and the noise), so
the candidates are compared through their first chunks: 16 steps x 12 action dims, each dim z-scored per task
(distribution.py). Winners finished stage k in both continuations, losers in neither; 1/2 candidates are left out.
Two sets of points: every split point (>= 1 winner and >= 1 loser), and the strict ones (>= 2 of each), which luck
alone produces far less often.
Per point:
  1-NN    leave-one-out: each winner or loser gets the label of its nearest other winner or loser; balanced accuracy
          (mean of the two recalls), against 500 label shuffles at that point
  split   purity of the best 2-group split of all 8 candidates (distribution.py) with respect to win / lose
  centre  distance to the centroid of the 8, winners minus losers (are winners the typical candidates?)
The 1-NN test is repeated on each action group alone and on the first and last 4 steps alone.
Also draws candidates_<task>.png: for the strict points with the largest effect, each candidate's commanded end-effector
path (cumulative end_effector_position command, top view), its gripper command and its arm/base mode. The chunks come
from the rollouts of the coarse row in use (data.py). Writes the PNGs to $CRIT_OUT.
usage: python -I candidates.py"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import OUT, TASKS  # noqa: E402
from distribution import DIM_GROUP, SPLITS, actions  # noqa: E402
from effect_by_event import points  # noqa: E402

rng = np.random.default_rng(0)


def nn_acc(X, lab):
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    np.fill_diagonal(D, np.inf)
    pred = lab[D.argmin(1)]
    return np.mean([np.mean(pred[lab == c] == c) for c in (True, False)])


def nn_test(X, lab, n=500):
    obs = nn_acc(X, lab)
    null = np.array([nn_acc(X, rng.permutation(lab)) for _ in range(n)])
    return obs, null.mean(), float((null >= obs).mean())


def best_split(X):
    C = X - X.mean(0)
    best, arg = -1.0, None
    for m in SPLITS:
        if m.all():
            continue
        b = m.sum() * ((X[m].mean(0) - X.mean(0)) ** 2).sum() + (~m).sum() * ((X[~m].mean(0) - X.mean(0)) ** 2).sum()
        if b > best:
            best, arg = b, m
    return arg


def main():
    pts = points()
    A = {}
    for p in pts:
        A[id(p)] = actions([np.load(f"{p['roll']}/{p['tag']}_t{p['t']}_c{i}_m0.npz") for i in range(8)])
    std = {t: np.concatenate([A[id(p)] for p in pts if p["task"] == t]).reshape(-1, 12).std(0) + 1e-6 for t in TASKS}
    crit = []
    for p in pts:
        per = p["y"].sum(1)
        W, L = per == 2, per == 0
        if W.any() and L.any():
            p["A"] = A[id(p)] / std[p["task"]]
            p["W"], p["L"], p["strict"] = W, L, W.sum() >= 2 and L.sum() >= 2
            crit.append(p)
    views = {"all": (slice(None), np.ones(12, bool)), "arm": (slice(None), DIM_GROUP == "arm"), "gripper": (slice(None), DIM_GROUP == "grip"),
             "base": (slice(None), DIM_GROUP == "base"), "mode": (slice(None), DIM_GROUP == "mode"),
             "steps 1-4": (slice(0, 4), np.ones(12, bool)), "steps 13-16": (slice(12, 16), np.ones(12, bool))}
    for name, sel in [("split points", lambda p: True), ("strict (>= 2 winners and >= 2 losers)", lambda p: p["strict"])]:
        g = [p for p in crit if sel(p)]
        print(f"=== {name}: {len(g)} points (" + ", ".join(f"{t} {sum(p['task'] == t for p in g)}" for t in TASKS) + ")")
        for v, (steps, dims) in views.items():
            res = []
            for p in g:
                keep = p["W"] | p["L"]
                X = p["A"][keep][:, steps][:, :, dims].reshape(keep.sum(), -1)
                if X.std() == 0:
                    continue
                res.append(nn_test(X, p["W"][keep]))
            r = np.array(res)
            print(f"  1-NN on {v:12s} n {len(r):3d}: balanced accuracy {r[:, 0].mean():.2f} vs {r[:, 1].mean():.2f} with shuffled labels; "
                  f"p < 0.05 at {np.mean(r[:, 2] < 0.05):.0%} of points (5 % by chance)")
        pur, cen = [], []
        for p in g:
            m = best_split(p["A"].reshape(8, -1))
            keep = p["W"] | p["L"]
            a = np.mean(m[keep] == p["W"][keep]); pur.append(max(a, 1 - a))
            d = np.linalg.norm(p["A"].reshape(8, -1) - p["A"].reshape(8, -1).mean(0), axis=1)
            cen.append(d[p["W"]].mean() - d[p["L"]].mean())
        null = []
        for p in g:
            m = best_split(p["A"].reshape(8, -1)); keep = p["W"] | p["L"]
            sh = [np.mean(m[keep] == rng.permutation(p["W"][keep])) for _ in range(200)]
            null.append(np.mean([max(a, 1 - a) for a in sh]))
        cen = np.array(cen)
        print(f"  best 2-group split vs win/lose: purity {np.mean(pur):.2f} vs {np.mean(null):.2f} with shuffled labels")
        print(f"  distance to the centroid, winners minus losers: median {np.median(cen):+.2f} (z units); winners more central at "
              f"{np.mean(cen < 0):.0%} of points")
    for task in TASKS:
        g = sorted([p for p in crit if p["strict"] and p["task"] == task], key=lambda p: (-p["cb"], p["tag"], p["t"]))[:6]
        if g:
            draw(task, g)


def draw(task, g):
    fig, axes = plt.subplots(3, len(g), figsize=(3.1 * len(g), 7.4), facecolor="#fcfcfb", squeeze=False,
                             gridspec_kw=dict(height_ratios=[3, 1, 1]))
    col = {2: "#2a78d6", 1: "#b9b8b3", 0: "#d95926"}
    for j, p in enumerate(g):
        raw = actions([np.load(f"{p['roll']}/{p['tag']}_t{p['t']}_c{i}_m0.npz") for i in range(8)])
        per = p["y"].sum(1).astype(int)
        for i in np.argsort(per):
            path = np.cumsum(raw[i, :, 0:3], 0)
            kw = dict(color=col[per[i]], lw=1.6 if per[i] != 1 else 1.0, alpha=0.95 if per[i] != 1 else 0.6)
            axes[0, j].plot(np.r_[0, path[:, 0]], np.r_[0, path[:, 1]], **kw)
            axes[0, j].plot(path[-1, 0], path[-1, 1], "o", ms=3.5, color=col[per[i]])
            axes[1, j].plot(raw[i, :, 6] + 0.04 * (i - 3.5), **kw)
            axes[2, j].plot(raw[i, :, 11] + 0.04 * (i - 3.5), **kw)
        axes[0, j].set_title(f"{p['tag'].split('_')[1]} t{p['t']} S{p['k']}\nP(done) {p['pd']:.2f}", fontsize=8.5, color="#0b0b0b")
        axes[0, j].set_aspect("equal", "datalim")
        for r, lab in enumerate(["commanded eef path, top view (action units)", "gripper_close", "control_mode (1 = base)"]):
            axes[r, j].tick_params(labelsize=6.5, colors="#52514e")
            if j == 0:
                axes[r, j].set_ylabel(lab, fontsize=7.5, color="#52514e")
        axes[2, j].set_xlabel("step in the chunk", fontsize=7, color="#52514e")
    fig.suptitle(f"{task}: candidates at the strongest critical points; blue = finished the stage in both continuations, "
                 f"orange = in neither, grey = one of two", fontsize=9.5, color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"candidates_{task}.png"), dpi=140, facecolor="#fcfcfb"); plt.close(fig)


if __name__ == "__main__":
    main()
