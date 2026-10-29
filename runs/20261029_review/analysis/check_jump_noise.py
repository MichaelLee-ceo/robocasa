"""make_dense_jobs.py marks an interval when two consecutive same-stage coarse points differ by |dV| >= 0.4 (16 rollouts
each, 8 candidates x 2). How many such pairs would noise alone give? For each consecutive same-stage pair, under "V did
not change" both points share q = the pair's mean V; each point is simulated as 8 candidates with p_i drawn around q
(between-candidate variance = the pair's mean var_b, Beta moment match; 0 = binomial) x 2 continuations.
Also the dense series: a jump between two dense points (8 rollouts each) compared with its own noise.
Read-only. usage: python -I check_jump_noise.py"""
import json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
CRIT = f"{R}/20261028_critical"
sys.path.insert(0, CRIT)
import numpy as np  # noqa: E402
from effect_by_event import RUNS, points  # noqa: E402

rng = np.random.default_rng(0)


def sim_point(q, vb, reps):
    if vb <= 1e-6 or q <= 0 or q >= 1:
        p = np.full((reps, 8), q)
    else:
        vb = min(vb, q * (1 - q) * 0.99)
        s = q * (1 - q) / vb - 1
        p = rng.beta(q * s, (1 - q) * s, size=(reps, 8))
    return (rng.random((reps, 8, 2)) < p[..., None]).mean((1, 2))


def main():
    by = {}
    for p in points():
        by.setdefault(p["tag"], []).append(p)
    pairs = []
    for g in by.values():
        g.sort(key=lambda p: p["t"])
        pairs += [(a, b) for a, b in zip(g[:-1], g[1:]) if a["k"] == b["k"]]
    obs = {t: 0 for t in RUNS}; exp0 = {t: 0.0 for t in RUNS}; expb = {t: 0.0 for t in RUNS}
    for a, b in pairs:
        t = a["task"]
        obs[t] += abs(b["pd"] - a["pd"]) >= 0.4
        q = (a["pd"] + b["pd"]) / 2
        vb = max((a["cb"] + b["cb"]) / 2, 0)
        reps = 4000
        exp0[t] += np.mean(np.abs(sim_point(q, 0, reps) - sim_point(q, 0, reps)) >= 0.4 - 1e-9)
        expb[t] += np.mean(np.abs(sim_point(q, vb, reps) - sim_point(q, vb, reps)) >= 0.4 - 1e-9)
    for t in RUNS:
        print(f"{t:21s} same-stage pairs {sum(a['task'] == t for a, _ in pairs):3d}: |dV| >= 0.4 observed {obs[t]:3d}; "
              f"expected from noise alone if V were flat at the pair's mean: {exp0[t]:4.1f} (binomial) / {expb[t]:4.1f} (with the candidate variance)")
    cc = json.load(open(f"{CRIT}/critical_chunks.json"))
    print("dense series: is the located jump larger than its own noise? z = dV / sqrt(sum of V(1-V)/n) (n: 8 dense, 16 coarse, exact at a finish)")
    for task in RUNS:
        z = []
        for r in cc:
            if r["task"] != task or not r["complete"]:
                continue
            s = {t: (v, n) for t, v, n in r["series"]}
            (va, na), (vb_, nb) = s[r["ta"]], s[r["tb"]]
            var = (va * (1 - va) / na if na else 0) + (vb_ * (1 - vb_) / nb if nb else 0)
            z.append(abs(r["jump"]) / np.sqrt(var) if var > 0 else np.inf)
        z = np.array(z)
        print(f"   {task:21s} intervals {len(z):3d}: located jump z < 2 in {int((z < 2).sum())}, 2-3 in {int(((z >= 2) & (z < 3)).sum())}, >= 3 in {int((z >= 3).sum())}")


if __name__ == "__main__":
    main()
