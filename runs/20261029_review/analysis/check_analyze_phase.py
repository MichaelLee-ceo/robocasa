"""analyze_phase.py takes one glob and lets later files replace earlier rows in sorted() order. It has no notion of the
re-collected dirs. This shows (a) which dir wins per point for the natural globs, and (b) the README's per-task numbers
(computed from run_SteamInMicrowave_v2 / run_PackIdenticalLunches alone) next to the same estimator on the rows in use
(effect_by_event.points(): RUNS -> FIX -> FIX3). Read-only. usage: python -I check_analyze_phase.py"""
import collections, glob, json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
sys.path.insert(0, f"{R}/20261028_critical")
from effect_by_event import points  # noqa: E402

rng = np.random.default_rng(0)


def load_like_analyze_phase(pattern):
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    rows = {}
    for p in files:
        for r in map(json.loads, open(p)):
            if "error" not in r:
                rows[(r["tag"], r["t"])] = (r, os.path.relpath(os.path.dirname(p), DATA))
    return rows


def cb(y):
    n = len(y); d0, d1 = y[:, 0] - y[:, 0].mean(), y[:, 1] - y[:, 1].mean()
    return (d0 * d1).sum() / (n - 1)


def summ(v):
    v = np.array(v)
    lo, hi = np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)], [2.5, 97.5])
    g = lambda x: 142 * np.sqrt(max(x, 0))
    return f"var_b {v.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] gain {g(v.mean()):.0f} [{g(lo):.0f}, {g(hi):.0f}]"


def main():
    for pat in ["run_SteamInMicrowave*/results_w*.jsonl", "run_PackIdenticalLunches*/results_w*.jsonl"]:
        rows = load_like_analyze_phase(pat)
        c = collections.Counter(d for _, d in rows.values())
        print(f"glob {pat!r}: {len(rows)} points; the row in use comes from {dict(c)}")
    cur = {(p["tag"], p["t"]): p for p in points()}
    main_hist = {}
    for line in open(f"{R}/20261028_critical/events.jsonl"):
        r = json.loads(line); main_hist[r["tag"]] = r["history"]
    for task, d in [("SteamInMicrowave", "run_SteamInMicrowave_v2"), ("PackIdenticalLunches", "run_PackIdenticalLunches")]:
        old = load_like_analyze_phase(f"{d}/results_w*.jsonl")
        print(f"--- {task}: README source ({d} only) vs the rows in use")
        groups = {"all": lambda k, el: True}
        if task == "SteamInMicrowave":
            groups.update({"S1 steps 160-240": lambda k, el: k == 1 and 160 <= el < 240})
        else:
            groups.update({"S0 steps 0-80": lambda k, el: k == 0 and el < 80, "S1 steps 240-400": lambda k, el: k == 1 and 240 <= el < 400})
        for lab, f in groups.items():
            vo, vn = [], []
            for key, (r, _) in old.items():
                k, t = r["k"], r["t"]
                h = main_hist[r["tag"]]
                el = t - (0 if k == 0 else h[k - 1][0])
                if f(k, el):
                    vo.append(cb(np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float)))
                    vn.append(cur[key]["cb"])
            print(f"   {lab:18s} n {len(vo):3d}  README source: {summ(vo)}   rows in use: {summ(vn)}")


if __name__ == "__main__":
    main()
