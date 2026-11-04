"""Question 2, step 2: which deployable signals (features.py) separate the critical / steerable chunks from the others?

Discovery set: the 64 points of confirm (32 critical chunks) + control (32 random chunks), 8 x 4 each.
  AUROC    P(signal at a positive point > signal at a negative point of the SAME task) (ties count 1/2). Pairs are
           formed within a task only, so a task's scale or base rate cannot fake a separation. 0.5 = no separation; below
           0.5 means the signal is LOWER at the positives. 95 % CI: 2000 bootstrap resamples of points within each
           (task, class) stratum.
  targets  crit: critical (confirm) vs random (control); steer: steerable (perm_p < 0.05) vs not, all 64;
           steer|crit: steerable vs not among the 32 critical chunks only; rho(var_b): Spearman correlation with var_b over
           the 64 (bootstrap CI over points).
Replication set: coarse, 768 uniform points with 8 x 2 (var_b per point is unbiased but very noisy). Per signal, with
  the signal turned into its within-task percentile:
  rho      Spearman correlation with var_b (all points; then only undecided points, 0 < V < 1 over the 16 rollouts)
  top25    the quarter of points with the highest signal (within task; ties broken at random): mean var_b and mean cv
           gain (a useless signal's quarter gets the coarse average); for a signal lower at the steerable points of the
           64, the lowest quarter
  combined every deployable signal in one ridge regression on coarse var_b (out-of-fold by episode), also scored on the 64
  stage    the stage index, and the stage's mean var_b on coarse as a prior scored on the 64
  undec    AUROC of the signal for "undecided" (0 < V < 1) vs decided: does it only detect that the stage is still open?
With ~30 signals, about 1-2 CIs per column exclude 0.5 by chance; only effects that hold on both sets count.
usage: python -I analyze.py > analyze.out"""
import json, os
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ("KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches")
GROUPS = {
    "candidate disagreement": ["var", "var_arm", "var_grip", "var_base", "var_mode", "bimodal", "minority", "pc1", "outlier",
                               "growth", "pos_rel", "grip_end", "act_mag", "base_frac", "close_frac"],
    "robot state / motion": ["gap", "arm_speed", "base_speed", "arm_moved", "base_moved", "gap_moved"],
    "contacts": ["touch", "touch16", "grasp"],
    "past events / stage": ["since_event", "k"],
    "references (not deployable)": ["t_stage", "t", "to_event"],
}
SIGNALS = [s for g in GROUPS.values() for s in g]
B = 2000
rng = np.random.default_rng(0)


def load():
    R = [json.loads(l) for l in open(os.path.join(HERE, "features.jsonl"))]
    for r in R:
        r["crit"] = r["ds"] == "confirm"
    return R


def auc(x, y, g):
    """within-group AUROC of x for y (bool), pooled over groups by pairs"""
    num = den = 0.0
    for gg in np.unique(g):
        m = g == gg
        pos, neg = x[m & y], x[m & ~y]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return num / den if den else np.nan


def strata_boot(y, g):
    """bootstrap index sets, resampling within each (group, class) stratum"""
    st = [np.flatnonzero((g == gg) & (y == c)) for gg in np.unique(g) for c in (True, False)]
    st = [s for s in st if len(s)]
    return [np.concatenate([rng.choice(s, len(s)) for s in st]) for _ in range(B)]


def spear(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else 0.0


def ci(vals):
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return lo, hi


def fmt_auc(x, y, g, idx):
    a = auc(x, y, g)
    lo, hi = ci([auc(x[i], y[i], g[i]) for i in idx])
    star = "*" if (lo > 0.5 or hi < 0.5) else " "
    return f"{a:.2f} [{lo:.2f},{hi:.2f}]{star}", a


def pct_within(R, s):
    """the signal as its percentile within the task (coarse points)"""
    out = np.zeros(len(R))
    for task in TASKS:
        m = np.array([r["task"] == task for r in R])
        x = np.array([float(r[s]) for r in R])[m]
        out[m] = (rankdata(x) - 0.5) / len(x)
    return out


def discovery(E):
    g = np.array([r["task"] for r in E])
    crit = np.array([r["crit"] for r in E])
    steer = np.array([r["steerable"] for r in E])
    steer05 = np.array([r["steer05"] for r in E])
    vb = np.array([r["var_b"] for r in E])
    i_crit, i_steer, i_steer05 = strata_boot(crit, g), strata_boot(steer, g), strata_boot(steer05, g)
    C = np.flatnonzero(crit)
    i_sc = strata_boot(steer[C], g[C])
    i_pts = [rng.integers(0, len(E), len(E)) for _ in range(B)]
    res = {}
    print(f"=== discovery: {len(E)} points (critical {crit.sum()}, random {(~crit).sum()}; steerable {steer.sum()} "
          f"[all critical], var_b >= 0.05: {steer05.sum()})")
    print("per task: " + "; ".join(f"{t}: critical {(crit & (g == t)).sum()}, steerable {(steer & (g == t)).sum()}, "
                                   f"steerable among critical {(steer & crit & (g == t)).sum()}" for t in TASKS))
    print("AUROC within task [95 % CI]; * = CI excludes 0.5")
    print(f"{'signal':12s} {'crit vs random':22s} {'steerable vs not':22s} {'steer|crit (n=32)':22s} {'rho(var_b), n=64':22s} {'(var_b>=.05 label)':s}")
    for grp, sigs in GROUPS.items():
        print(f"--- {grp}")
        for s in sigs:
            x = np.array([float(r[s]) for r in E])
            a1, v1 = fmt_auc(x, crit, g, i_crit)
            a2, v2 = fmt_auc(x, steer, g, i_steer)
            a3, v3 = fmt_auc(x[C], steer[C], g[C], i_sc)
            a4, _ = fmt_auc(x, steer05, g, i_steer05)
            rho = spear(x, vb)
            lo, hi = ci([spear(x[i], vb[i]) for i in i_pts])
            rs = f"{rho:+.2f} [{lo:+.2f},{hi:+.2f}]" + ("*" if lo > 0 or hi < 0 else " ")
            print(f"{s:12s} {a1:22s} {a2:22s} {a3:22s} {rs:22s} {a4}")
            res[s] = dict(crit=v1, steer=v2, steer_crit=v3, rho=rho, star=[a1[-1] == "*", a2[-1] == "*", a3[-1] == "*"])
    print("per task, for the signals whose CI excludes 0.5 above (KettleBoiling / SteamInMicrowave / PackIdenticalLunches; "
          "n positives vs negatives in brackets)")
    for s in SIGNALS:
        for j, (lab, y, sel) in enumerate([("crit vs random", crit, np.ones(len(E), bool)), ("steerable vs not", steer, np.ones(len(E), bool)),
                                           ("steer|crit", steer, crit)]):
            if res[s]["star"][j]:
                x = np.array([float(r[s]) for r in E])
                parts = [f"{auc(x[sel & (g == t)], y[sel & (g == t)], g[sel & (g == t)]):.2f} ({(y & sel & (g == t)).sum()}v{(~y & sel & (g == t)).sum()})"
                         for t in TASKS]
                print(f"  {s:12s} {lab:17s} " + "  ".join(parts))
    return res


def stage_table(E, Co):
    print("=== stage: points per (task, stage) — critical / random / steerable of the 64; coarse: n, undecided share, mean var_b")
    for t in TASKS:
        for k in sorted({r["k"] for r in Co if r["task"] == t}):
            e = [r for r in E if r["task"] == t and r["k"] == k]
            c = [r for r in Co if r["task"] == t and r["k"] == k]
            und = np.mean([0 < r["V"] < 1 for r in c])
            print(f"  {t:21s} S{k} ({e[0]['kind'] if e else c[0]['kind']:7s}) critical {sum(r['crit'] for r in e):2d} random "
                  f"{sum(not r['crit'] for r in e):2d} steerable {sum(r['steerable'] for r in e):2d} | coarse n {len(c):3d} "
                  f"undecided {und:.2f} mean var_b {np.mean([r['var_b'] for r in c]):+.4f}")
    # the stage as a predictor: score each of the 64 points by its (task, stage)'s mean var_b on coarse (independent data)
    prior = {}
    for r in Co:
        prior.setdefault((r["task"], r["k"]), []).append(r["var_b"])
    sc = np.array([np.mean(prior[(r["task"], r["k"])]) for r in E]); g = np.array([r["task"] for r in E])
    for lab in ("crit", "steerable"):
        y = np.array([r[lab] for r in E])
        a, _ = fmt_auc(sc, y, g, strata_boot(y, g))
        print(f"  stage prior (the stage's mean var_b on coarse) as a predictor of {lab}: AUROC {a}")


def top_quarter(x, g):
    """the 25 % of points with the highest x within each task; ties broken at random (fixed seed), so a binary signal
    selects its "on" points first and fills or thins the quarter at random"""
    q = np.zeros(len(x), bool)
    tie = np.random.default_rng(1).random(len(x))
    for t in TASKS:
        m = np.flatnonzero(g == t)
        order = m[np.lexsort((tie[m], x[m]))[::-1]]
        q[order[:int(round(0.25 * len(m)))]] = True
    return q


def replication(Co, top):
    print(f"=== replication on coarse: {len(Co)} points, 8 x 2; undecided {sum(0 < r['V'] < 1 for r in Co)}")
    vb = np.array([r["var_b"] for r in Co]); cv = np.array([r["cv"] for r in Co])
    g = np.array([r["task"] for r in Co]); und = np.array([0 < r["V"] < 1 for r in Co])
    i_pts = [rng.integers(0, len(Co), len(Co)) for _ in range(B)]
    U = np.flatnonzero(und)
    i_und = [rng.choice(U, len(U)) for _ in range(B)]
    i_ud = strata_boot(und, g)
    print(f"all coarse: mean var_b {vb.mean():+.4f}, mean cv gain {100 * cv.mean():+.1f} pts (what a useless signal's top quarter gets)")
    print(f"{'signal':12s} {'dir':3s} {'rho(var_b) all':22s} {'rho(var_b) undecided':22s} {'top25: mean var_b':26s} "
          f"{'cv gain pts':20s} {'AUROC undecided':20s}")
    for s, sign in top:
        x = sign * pct_within(Co, s)
        rho = spear(x, vb); lo, hi = ci([spear(x[i], vb[i]) for i in i_pts])
        ru = spear(x[U], vb[U]); ulo, uhi = ci([spear(x[i], vb[i]) for i in i_und])
        q = top_quarter(x, g)
        mv = vb[q].mean(); mlo, mhi = ci([vb[i][q[i]].mean() for i in i_pts])
        cg = 100 * cv[q].mean(); clo, chi = ci([100 * cv[i][q[i]].mean() for i in i_pts])
        a, _ = fmt_auc(x, und, g, i_ud)
        print(f"{s:12s} {'+' if sign > 0 else '-':3s} {rho:+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 or hi < 0 else ' '}   "
              f"{ru:+.2f} [{ulo:+.2f},{uhi:+.2f}]{'*' if ulo > 0 or uhi < 0 else ' '}   {mv:+.4f} [{mlo:+.4f},{mhi:+.4f}]   "
              f"{cg:+5.1f} [{clo:+.1f},{chi:+.1f}]   {a}")


DEPLOY = [s for g, sigs in GROUPS.items() if not g.startswith("references") for s in sigs]


def pct_ref(R, ref, s):
    """the signal as its percentile among the coarse points of the same task (ref)"""
    out = np.zeros(len(R))
    for task in TASKS:
        xr = np.sort([float(r[s]) for r in ref if r["task"] == task])
        m = np.array([r["task"] == task for r in R])
        x = np.array([float(r[s]) for r in R])[m]
        out[m] = (np.searchsorted(xr, x, "left") + np.searchsorted(xr, x, "right")) / (2 * len(xr))
    return out


def ridge(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.linalg.solve(Z.T @ Z + lam * np.diag([0] + [1] * X.shape[1]), Z.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w


def combined(E, Co):
    """every deployable signal (within-task percentiles vs coarse) in one ridge regression on coarse var_b (unbiased per
    point, so E[var_b | signals] is learned without bias). Out-of-fold on coarse (5 folds by episode), then fit on all of
    coarse and scored on the 64."""
    X = np.c_[[pct_ref(Co, Co, s) for s in DEPLOY]].T
    vb = np.array([r["var_b"] for r in Co]); cv = np.array([r["cv"] for r in Co]); g = np.array([r["task"] for r in Co])
    tags = sorted({r["tag"] for r in Co}); fold = {t: i % 5 for i, t in enumerate(np.random.default_rng(2).permutation(tags))}
    f = np.array([fold[r["tag"]] for r in Co]); pred = np.zeros(len(Co))
    for k in range(5):
        pred[f == k] = ridge(X[f != k], vb[f != k])(X[f == k])
    i_pts = [rng.integers(0, len(Co), len(Co)) for _ in range(B)]
    q = top_quarter(pred, g)
    rho = spear(pred, vb); lo, hi = ci([spear(pred[i], vb[i]) for i in i_pts])
    mv = vb[q].mean(); mlo, mhi = ci([vb[i][q[i]].mean() for i in i_pts])
    cg = 100 * cv[q].mean(); clo, chi = ci([100 * cv[i][q[i]].mean() for i in i_pts])
    print(f"=== all {len(DEPLOY)} deployable signals combined (ridge on coarse var_b)")
    print(f"  coarse, out-of-fold by episode: rho(var_b) {rho:+.2f} [{lo:+.2f},{hi:+.2f}]; top quarter mean var_b {mv:+.4f} "
          f"[{mlo:+.4f},{mhi:+.4f}] (all {vb.mean():+.4f}), cv gain {cg:+.1f} [{clo:+.1f},{chi:+.1f}] pts (all {100 * cv.mean():+.1f})")
    sE = ridge(X, vb)(np.c_[[pct_ref(E, Co, s) for s in DEPLOY]].T)
    gE = np.array([r["task"] for r in E])
    for lab in ("crit", "steerable"):
        y = np.array([r[lab] for r in E])
        a, _ = fmt_auc(sE, y, gE, strata_boot(y, gE))
        print(f"  fit on coarse, scored on the 64: AUROC {lab} {a}")


def main():
    R = load()
    E = [r for r in R if r["ds"] in ("confirm", "control")]
    Co = [r for r in R if r["ds"] == "coarse"]
    res = discovery(E)
    stage_table(E, Co)
    combined(E, Co)
    # replicate every signal; the direction is the one seen on the discovery set (steerable vs not)
    top = [(s, 1 if res[s]["steer"] >= 0.5 else -1) for s in SIGNALS if s != "k"]
    replication(Co, top)


if __name__ == "__main__":
    main()
