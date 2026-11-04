"""Step 3 of README.md: does DiVeR's u_t (candidate dispersion in GR00T's action-head feature space) mark our critical /
steerable chunks?

u_t = mean over the 1024 coordinates of Var_k[z_k], z_k = candidate k's DiT output at the 16 action positions of the last
denoising step, averaged over the positions (DiVeR Eq. 8; population variance). Variants:
  u8   the 8 labelled candidates (seeds k < 8)          u32  all 32 seeds (confirm / control only; DiVeR's K)
  utok the same dispersion per action token, then averaged over tokens (not DiVeR's; does averaging z lose it?)
  uz   u8 standardized within its main run, by the mean and std of u8 over that run's coarse points (DiVeR Eq. 9's
       per-trajectory standardization)
Reference row: var, the raw-action candidate variance of the signals study (../20261029_signals/features.jsonl).
Tables (helpers and labels from ../20261029_signals/analyze.py and features.jsonl, same bootstrap):
  discovery  the 64 confirm / control points: within-task AUROC for critical vs random, steerable vs not, steerable
             among critical, and Spearman with var_b
  replication the 768 coarse points: Spearman with var_b (all; undecided, 0 < V < 1), and AUROC for undecided vs decided
  temporal   per critical chunk, the dense points of the same main run within +-64 steps: the percentile of u at the
             critical chunk among them (0.5 = no preference), and where the largest u sits relative to it
usage: python -I analyze_u.py   (groot python; reads feats/)
"""
import importlib.util, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "20261028_critical"))
import data  # noqa: E402

_spec = importlib.util.spec_from_file_location("signals", os.path.join(HERE, "..", "20261029_signals", "analyze.py"))
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)
FEATS = os.path.join(HERE, "feats")
WIN = 64


def u_of(task, ds, tag, t):
    p = os.path.join(FEATS, task, ds, f"{tag}_t{t}.npz")
    if not os.path.exists(p):
        return None
    f = np.load(p)["feat"].astype(np.float64)  # K x 16 x 1024
    z = f.mean(1)
    out = dict(u8=z[:8].var(0).mean(), utok=f[:8].var(0).mean())
    if len(z) >= 32:
        out["u32"] = z[:32].var(0).mean()
    return out


def coarse():
    """coarse points with their u, and per main run the mean / std of u8 over its coarse points"""
    C = [r for r in S.load() if r["ds"] == "coarse"]
    for r in C:
        r.update(u_of(r["task"], "coarse", r["tag"], r["t"]) or {})
    C = [r for r in C if "u8" in r]
    by = {}
    for r in C:
        by.setdefault(r["tag"], []).append(r["u8"])
    norm = {tag: (np.mean(v), np.std(v)) for tag, v in by.items() if len(v) >= 3 and np.std(v) > 0}
    for r in C:
        if r["tag"] in norm:
            r["uz"] = (r["u8"] - norm[r["tag"]][0]) / norm[r["tag"]][1]
    return C, norm


def replication(C):
    g = np.array([r["task"] for r in C]); vb = np.array([r["var_b"] for r in C]); V = np.array([r["V"] for r in C])
    und = (V > 0) & (V < 1)
    print(f"=== replication: {len(C)} coarse points (undecided {und.sum()})")
    print(f"{'signal':8s} {'rho(var_b) all':22s} {'rho(var_b) undecided':22s} {'undecided vs decided':22s}")
    idx_a = S.strata_boot(und, g)
    idx_u = S.strata_boot(np.ones(und.sum(), bool), g[und])
    for s in ("u8", "utok", "uz", "var"):
        sel = np.array([s in r for r in C])
        if not sel.all():
            continue
        x = np.array([float(r[s]) for r in C])
        r_all = S.spear(x, vb); lo, hi = S.ci([S.spear(x[i], vb[i]) for i in idx_a])
        r_und = S.spear(x[und], vb[und]); lo2, hi2 = S.ci([S.spear(x[und][i], vb[und][i]) for i in idx_u])
        a, _ = S.fmt_auc(x, und, g, idx_a)
        print(f"{s:8s} {r_all:+.2f} [{lo:+.2f},{hi:+.2f}]        {r_und:+.2f} [{lo2:+.2f},{hi2:+.2f}]        {a}")


def discovery(norm):
    E = [r for r in S.load() if r["ds"] in ("confirm", "control")]
    for r in E:
        r.update(u_of(r["task"], r["ds"], r["tag"], r["t"]) or {})
        if "u8" in r and r["tag"] in norm:
            r["uz"] = (r["u8"] - norm[r["tag"]][0]) / norm[r["tag"]][1]
    E = [r for r in E if "u8" in r]
    g = np.array([r["task"] for r in E]); crit = np.array([r["crit"] for r in E])
    steer = np.array([r["steerable"] for r in E]); vb = np.array([r["var_b"] for r in E])
    print(f"=== discovery: {len(E)} points (critical {crit.sum()}, random {(~crit).sum()}, steerable {steer.sum()})")
    print(f"{'signal':8s} {'crit vs random':22s} {'steerable vs not':22s} {'steer|crit':22s} {'rho(var_b)':22s}")
    idx_c, idx_s = S.strata_boot(crit, g), S.strata_boot(steer, g)
    m = crit
    idx_sc = S.strata_boot(steer[m], g[m])
    for s in ("u8", "u32", "utok", "uz", "var"):
        if not all(s in r for r in E):
            print(f"{s:8s} missing at {sum(s not in r for r in E)} points")
            continue
        x = np.array([float(r[s]) for r in E])
        a1, _ = S.fmt_auc(x, crit, g, idx_c)
        a2, _ = S.fmt_auc(x, steer, g, idx_s)
        a3, _ = S.fmt_auc(x[m], steer[m], g[m], idx_sc)
        rho = S.spear(x, vb)
        lo, hi = S.ci([S.spear(x[i], vb[i]) for i in idx_c])
        print(f"{s:8s} {a1:22s} {a2:22s} {a3:22s} {rho:+.2f} [{lo:+.2f},{hi:+.2f}]")
    if all("u32" in r for r in E):
        print(f"u8 vs u32 Spearman over the points: {S.spear(np.array([r['u8'] for r in E]), np.array([r['u32'] for r in E])):+.2f}")
    return E


def temporal():
    print(f"=== temporal: u at each critical chunk vs the dense points of its main run within +-{WIN} steps")
    pct, where = [], []
    for task in data.TASKS:
        D = data.rows(task, "dense")
        for (tag, tc), r in sorted(data.rows(task, "confirm").items()):
            uc = u_of(task, "confirm", tag, tc)
            near = [(t, u_of(task, "dense", tag, t), float(data.outcomes(D[(tag, t)]).mean()))
                    for (tg, t) in D if tg == tag and t != tc and abs(t - tc) <= WIN]
            near = [(t, u["u8"], v) for t, u, v in near if u]
            if uc is None or not near:
                continue
            us = np.array([u for _, u, _ in near])
            pct.append(((us < uc["u8"]).sum() + 0.5 * (us == uc["u8"]).sum()) / len(us))
            seq = sorted(near + [(tc, uc["u8"], float(data.outcomes(r)[:, 0].mean()))])
            tmax = max(seq, key=lambda x: x[1])[0]
            where.append((tmax - tc) // 16)
            print(f"  {tag} t{tc}: " + " ".join(f"{'*' if t == tc else ''}{t}:u{u:.3g}/V{v:.2f}" for t, u, v in seq))
    if pct:
        pct = np.array(pct)
        print(f"critical chunks with dense neighbours: {len(pct)}; mean percentile of u at the critical chunk {pct.mean():.2f} "
              f"(0.5 = none); largest u at the critical chunk in {sum(w == 0 for w in where)}/{len(where)}, "
              f"one chunk before {sum(w == -1 for w in where)}, one after {sum(w == 1 for w in where)}")


if __name__ == "__main__":
    C, norm = coarse()
    discovery(norm)
    replication(C)
    temporal()
