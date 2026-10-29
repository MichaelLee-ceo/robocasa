"""Task 1: V(t) per main-run episode and stage, swing flags, and the binomial-noise null.

V(t) = mean(done) over the rollouts of the point (16 at coarse points, 8 at dense / dense2 points). Also V_ok =
mean(done and prev_ok). Series = the points of one (episode, stage k), sorted by t.

Flags (both directions):
  DIP   : a maximal run of >= 1 consecutive points with V <= 0.2, with V >= 0.8 at the point right before and right after
  SPIKE : a maximal run with V >= 0.8, with V <= 0.2 right before and right after
  JR    : consecutive (a, b, c) with |V_b - V_a| >= 0.75 and V_c - V_b of the opposite sign with |V_c - V_b| >= 0.5
Binomial null:
  (1) per triple / run, the worst-case p-value: max over a common true p of P(pattern at least as extreme);
  (2) expected number of flags if V were constant over each triple (p = the pooled rate of the triple);
  (3) parametric bootstrap with a monotone truth per series (weighted isotonic fit, increasing or decreasing,
      whichever fits better): no real swing exists, so every flag is noise. 2000 replicates.
Also checks k == k_replay == the main run's stage at t (history semantics) for every row.
Writes vseries.json, flags.json. usage: python vseries.py"""
import json, os, sys
import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, TASKS3, coarse, dense, rolls, main_runs, stage_at  # noqa: E402

HI, LO, JUMP, REV = 0.8, 0.2, 0.75, 0.5


def build():
    mr = main_runs()
    pts, kcheck = [], []
    for task in TASKS3:
        for src, rows in (("coarse", coarse(task)), ("dense", dense(task))):
            for (tag, t), r in rows.items():
                y = [x["done"] for _, _, x in rolls(r)]
                yo = [x["done"] and x["prev_ok"] for _, _, x in rolls(r)]
                k_hist = stage_at(mr[tag]["history"], t)
                kcheck.append(dict(task=task, tag=tag, t=t, k=r["k"], k_replay=r["k_replay"], k_hist=k_hist, dir=r["_dir"]))
                pts.append(dict(task=task, tag=tag, t=t, k=r["k"], n=len(y), x=int(sum(y)), V=float(np.mean(y)),
                                V_ok=float(np.mean(yo)), src=src, dir=r["_dir"], hidden=r.get("hidden"),
                                replaced=r.get("_replaced", [])))
    return pts, kcheck


def series(pts):
    by = {}
    for p in pts:
        by.setdefault((p["task"], p["tag"], p["k"]), []).append(p)
    for v in by.values():
        v.sort(key=lambda p: p["t"])
    return by


def flags_of(s, key="V"):
    """flags in one series (list of points sorted by t)."""
    out = []
    V = [p[key] for p in s]
    n = len(s)
    # runs of low / high points
    for kind, inside, outside in (("DIP", lambda v: v <= LO, lambda v: v >= HI), ("SPIKE", lambda v: v >= HI, lambda v: v <= LO)):
        i = 1
        while i < n - 1:
            if inside(V[i]) and outside(V[i - 1]):
                j = i
                while j < n and inside(V[j]):
                    j += 1
                if j < n and outside(V[j]):
                    out.append(dict(kind=kind, idx=list(range(i - 1, j + 1)), run=j - i))
                i = j
            else:
                i += 1
    for i in range(n - 2):
        d1, d2 = V[i + 1] - V[i], V[i + 2] - V[i + 1]
        if abs(d1) >= JUMP and d1 * d2 < 0 and abs(d2) >= REV:
            out.append(dict(kind="JR", idx=[i, i + 1, i + 2], run=1))
    return out


def worst_p(s, f):
    """max over a common p of P(outer points at least as high (DIP) / low (SPIKE) as observed and inner points at least
    as low / high as observed), for the points of flag f. For JR: P(|jump| >= observed and reversal >= observed) is
    approximated by the same tail product on (a, b, c)."""
    pts = [s[i] for i in f["idx"]]
    if f["kind"] == "JR":
        a, b, c = pts
        down = b["V"] < a["V"]
        f2 = dict(kind="DIP" if down else "SPIKE")
        pts = [a, b, c]
    else:
        down = f["kind"] == "DIP"
    best = 0.0
    for p in np.linspace(0.001, 0.999, 999):
        pr = 1.0
        for q, pt in enumerate(pts):
            outer = q in (0, len(pts) - 1)
            hi_side = outer if down else not outer
            if hi_side:  # P(X >= x)
                pr *= binom.sf(pt["x"] - 1, pt["n"], p)
            else:  # P(X <= x)
                pr *= binom.cdf(pt["x"], pt["n"], p)
        best = max(best, pr)
    return best


def p_flag_triple_const(na, nb, nc, p):
    """P(a 3-point DIP or SPIKE or JR flag) if all three points have true rate p (exact, by enumeration)."""
    pa, pb, pc = binom.pmf(np.arange(na + 1), na, p), binom.pmf(np.arange(nb + 1), nb, p), binom.pmf(np.arange(nc + 1), nc, p)
    va, vb, vc = np.arange(na + 1) / na, np.arange(nb + 1) / nb, np.arange(nc + 1) / nc
    A, B, C = np.meshgrid(va, vb, vc, indexing="ij")
    P = pa[:, None, None] * pb[None, :, None] * pc[None, None, :]
    dip = (A >= HI) & (B <= LO) & (C >= HI)
    spike = (A <= LO) & (B >= HI) & (C <= LO)
    d1, d2 = B - A, C - B
    jr = (np.abs(d1) >= JUMP) & (d1 * d2 < 0) & (np.abs(d2) >= REV)
    return float(P[dip].sum()), float(P[spike].sum()), float(P[jr].sum())


def isotonic(y, w, increasing=True):
    y, w = (np.asarray(y, float), np.asarray(w, float)) if increasing else (-np.asarray(y, float), np.asarray(w, float))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, c2 = out.pop(); v1, w1, c1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    fit = np.concatenate([[v] * c for v, _, c in out])
    return fit if increasing else -fit


def monotone_truth(s):
    y = np.array([p["V"] for p in s]); w = np.array([p["n"] for p in s], float)
    best = None
    for inc in (True, False):
        f = isotonic(y, w, inc)
        sse = float((w * (y - f) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, f)
    return np.clip(best[1], 0, 1)


def main():
    pts, kcheck = build()
    bad_k = [c for c in kcheck if not (c["k"] == c["k_replay"] == c["k_hist"])]
    print(f"rows checked for k: {len(kcheck)}; k != k_replay or != main-run stage at t: {len(bad_k)}")
    for c in bad_k[:20]:
        print("   ", c)
    by = series(pts)
    rng = np.random.default_rng(0)
    allflags = []
    summary = {}
    for task in TASKS3:
        keys = [k for k in by if k[0] == task]
        npts = sum(len(by[k]) for k in keys)
        ntrip = sum(max(len(by[k]) - 2, 0) for k in keys)
        obs = {"DIP": 0, "SPIKE": 0, "JR": 0}
        exp_const = {"DIP": 0.0, "SPIKE": 0.0, "JR": 0.0}
        for key in keys:
            s = by[key]
            for f in flags_of(s):
                obs[f["kind"]] += 1
                f.update(task=task, tag=key[1], k=key[2], worst_p=worst_p(s, f),
                         points=[dict(t=s[i]["t"], V=s[i]["V"], x=s[i]["x"], n=s[i]["n"], src=s[i]["src"], dir=s[i]["dir"],
                                      hidden=s[i]["hidden"], V_ok=s[i]["V_ok"]) for i in f["idx"]],
                         context=[(s[i]["t"], round(s[i]["V"], 2), s[i]["n"]) for i in range(max(0, f["idx"][0] - 3), min(len(s), f["idx"][-1] + 4))])
                allflags.append(f)
            for i in range(len(s) - 2):
                a, b, c = s[i], s[i + 1], s[i + 2]
                p = (a["x"] + b["x"] + c["x"]) / (a["n"] + b["n"] + c["n"])
                d, sp, jr = p_flag_triple_const(a["n"], b["n"], c["n"], p)
                exp_const["DIP"] += d; exp_const["SPIKE"] += sp; exp_const["JR"] += jr
        # parametric bootstrap, monotone truth
        truths = {key: monotone_truth(by[key]) for key in keys}
        boot = {"DIP": [], "SPIKE": [], "JR": []}
        for _ in range(2000):
            cnt = {"DIP": 0, "SPIKE": 0, "JR": 0}
            for key in keys:
                s = by[key]
                if len(s) < 3:
                    continue
                n = np.array([p["n"] for p in s])
                x = rng.binomial(n, truths[key])
                sim = [dict(V=xi / ni) for xi, ni in zip(x, n)]
                for f in flags_of(sim):
                    cnt[f["kind"]] += 1
            for kk in cnt:
                boot[kk].append(cnt[kk])
        summary[task] = dict(series=len(keys), points=npts, triples=ntrip, observed=obs,
                             expected_const_p={k: round(v, 3) for k, v in exp_const.items()},
                             boot_monotone={k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)),
                                                    p_ge_obs=float(np.mean(np.array(v) >= obs[k]))) for k, v in boot.items()})
        print(task, json.dumps(summary[task]))
    json.dump(dict(points=pts, kcheck_bad=bad_k, n_kcheck=len(kcheck)), open(os.path.join(HERE, "vseries.json"), "w"), indent=0)
    json.dump(dict(summary=summary, flags=allflags), open(os.path.join(HERE, "flags.json"), "w"), indent=0)
    for f in sorted(allflags, key=lambda f: (f["task"], f["tag"], f["k"], f["points"][0]["t"])):
        pp = " ".join(f"{q['t']}:{q['x']}/{q['n']}({q['hidden'][8:] if q['hidden'] and len(q['hidden']) > 7 else 'v1'})" for q in f["points"])
        print(f"{f['kind']:5s} {f['tag']:32s} S{f['k']} run={f['run']} p_worst={f['worst_p']:.1e}  {pp}")


if __name__ == "__main__":
    main()
