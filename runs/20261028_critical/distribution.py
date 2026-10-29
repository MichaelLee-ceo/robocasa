"""Does the shape of the 8 candidates' distribution show where the chunk matters? (source A, done properly)

Each candidate's first chunk is flattened to 16 steps x 12 action dims (end_effector_position 3, end_effector_rotation 3,
gripper_close 1, base_motion 4, control_mode 1). Every dim is z-scored with its std over all saved candidate chunks of
the task, so dims of different units weigh alike. Per decision point, from the 8 normalized chunks X (8 x 192):
  var      total variance: mean over the 192 coordinates of the variance across the 8
  var_arm / var_grip / var_base / var_mode   the same over each action group's coordinates
  bimodal  share of the total sum of squares explained by the best split of the 8 into two groups (exhaustive over the
           127 splits): near 1 when the candidates form two distinct options
  minority size of the smaller group of that best split (1 = a lone outlier, 4 = two equal options)
  pc1      share of the variance along the first principal direction: the candidates differ along one direction
  outlier  largest distance to the centroid / the median distance
  growth   variance over the last 4 steps / over the first 4 steps of the chunk
Reported against the per-point effect (between-candidate variance of P(done), as effect_by_event.py): Spearman
correlation with a bootstrap CI over points, and the best-of-8 gain by quartile (effect_by_event.summary). The chunks
come from the rollouts of the coarse row in use (data.py). usage: python -I distribution.py"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import TASKS  # noqa: E402
from effect_by_event import points, summary  # noqa: E402

KEYS = ["action.end_effector_position", "action.end_effector_rotation", "action.gripper_close", "action.base_motion", "action.control_mode"]
DIM_GROUP = np.array(["arm"] * 6 + ["grip"] + ["base"] * 4 + ["mode"])  # the 12 action dims, in KEYS order
SPLITS = [np.array([i in s for i in range(8)]) for r in range(1, 5) for s in itertools.combinations(range(8), r) if 0 in s or r < 4]
rng = np.random.default_rng(0)


def shape(X, groups):
    C = X - X.mean(0)
    v = (C ** 2).sum(0) / 7
    out = dict(var=float(v.mean()), **{f"var_{g}": float(v[groups == g].mean()) for g in ("arm", "grip", "base", "mode")})
    ss = (C ** 2).sum()
    best, mino = 0.0, 0
    for m in SPLITS:
        if m.all():
            continue
        b = m.sum() * ((X[m].mean(0) - X.mean(0)) ** 2).sum() + (~m).sum() * ((X[~m].mean(0) - X.mean(0)) ** 2).sum()
        if b > best:
            best, mino = b, int(min(m.sum(), (~m).sum()))
    out.update(bimodal=float(best / ss) if ss > 0 else 0.0, minority=mino)
    sv = np.linalg.svd(C, compute_uv=False) ** 2
    out["pc1"] = float(sv[0] / sv.sum()) if sv.sum() > 0 else 0.0
    d = np.linalg.norm(C, axis=1)
    out["outlier"] = float(d.max() / (np.median(d) + 1e-9))
    return out


def actions(chunks):
    """8 x 16 x 12: the candidates' first chunks, all action dims"""
    return np.concatenate([np.stack([c[k][0] for c in chunks]).astype(float) for k in KEYS], 2)


def spearman(x, y):
    from scipy.stats import rankdata  # average ranks: many points tie at exactly 0
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else 0.0


def main():
    pts = points()
    raw = {}
    for p in pts:
        raw[(p["tag"], p["t"])] = [np.load(f"{p['roll']}/{p['tag']}_t{p['t']}_c{i}_m0.npz") for i in range(8)]
    std = {}
    for task in TASKS:  # per-task std of every (step, dim) coordinate's dim, over all saved candidate chunks
        A = np.concatenate([actions(raw[(p["tag"], p["t"])]) for p in pts if p["task"] == task])  # n x 16 x 12
        std[task] = A.reshape(-1, A.shape[-1]).std(0) + 1e-6
    for p in pts:
        A = actions(raw[(p["tag"], p["t"])]) / std[p["task"]]  # 8 x 16 x 12, normalized
        p.update(shape(A.reshape(8, -1), np.tile(DIM_GROUP, 16)))
        v = A.var(0)  # 16 x 12
        p["growth"] = float(v[-4:].mean() / (v[:4].mean() + 1e-9))
    feats = ["var", "var_arm", "var_grip", "var_base", "var_mode", "bimodal", "minority", "pc1", "outlier", "growth"]
    print(f"{len(pts)} decision points; Spearman correlation with the per-point effect (95 % bootstrap CI), pooled and per task")
    for f in feats:
        line = []
        for scope in ["all", *TASKS]:
            g = [p for p in pts if scope == "all" or p["task"] == scope]
            x, y = np.array([p[f] for p in g]), np.array([p["cb"] for p in g])
            bs = [spearman(x[i], y[i]) for i in (rng.integers(0, len(g), len(g)) for _ in range(500))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            line.append(f"{scope[:5]} {spearman(x, y):+.2f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"  {f:9s} " + "   ".join(line))
    for f in ["var_grip", "var_mode"]:  # mostly exactly 0 (all 8 agree): zero vs non-zero, per task
        print(f"--- {f}: the 8 candidates agree exactly (0) or not, per task")
        for task in TASKS:
            for lab, cond in [("agree", lambda v: v == 0), ("differ", lambda v: v > 0)]:
                print(f"  {task[:5]} {lab:6s}", summary([p for p in pts if p["task"] == task and cond(p[f])]))
    for f in ["var", "bimodal", "pc1"]:
        print(f"--- {f} quartiles within each task (so tasks of different scale mix fairly)")
        q = {}
        for task in TASKS:
            g = [p for p in pts if p["task"] == task]
            e = np.quantile([p[f] for p in g], [0.25, 0.5, 0.75])
            for p in g:
                q[id(p)] = int(np.searchsorted(e, p[f], side="right"))
        for b in range(4):
            print(f"  Q{b + 1}", summary([p for p in pts if q[id(p)] == b]))


if __name__ == "__main__":
    main()
