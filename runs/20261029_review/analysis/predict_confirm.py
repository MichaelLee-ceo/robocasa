"""What after_refine.sh will do once the last dense2 worker ends: dense_analysis.py (no --refine), then make_confirm_jobs.py
for SteamInMicrowave and PackIdenticalLunches. Re-runs that logic in memory on the current files (nothing is written)
and reports, per predicted confirmation point:
  - whether (tag, ta) is also a coarse or a dense point: collect.py's seed does not depend on N, M or the run, so the
    confirmation's continuation 0 (and 1 at a coarse point) gets the same noise seeds as the rollouts that located it;
  - whether the interval is one of the stale ones (dense_intervals.json was written before the s4 t160 re-collection).
Read-only. usage: python -I predict_confirm.py"""
import glob, json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
CRIT = f"{R}/20261028_critical"
sys.path.insert(0, CRIT)
import numpy as np  # noqa: E402
from effect_by_event import RUNS, points  # noqa: E402
from dense_analysis import dense_v  # noqa: E402
from make_dense_jobs import intervals  # noqa: E402


def critical_rows(iv, coarse, dense):
    """dense_analysis.main()'s loop, unchanged (without the events)"""
    step = {"KettleBoiling": 16, "SteamInMicrowave": 32, "PackIdenticalLunches": 32}
    rows = []
    for i in iv:
        ts = list(range(i["t0"], i["t1"], step[i["task"]])) + [i["t1"]]
        vs = []
        for t in ts:
            if t == i["t1"] and i["kind"] == "rise_to_done":
                vs.append((t, 1.0, 0))
            elif (i["tag"], t) in coarse and t in (i["t0"], i["t1"]):
                vs.append((t, *coarse[(i["tag"], t)]))
            elif (i["tag"], t) in dense:
                vs.append((t, *dense[(i["tag"], t)]))
        for t in range(i["t0"] + 16, i["t1"], 16):
            if t not in ts and (i["tag"], t) in dense:
                vs.append((t, *dense[(i["tag"], t)]))
        vs.sort()
        done = len(vs) == len(ts) or len(vs) > len(ts)
        if len(vs) < 2:
            continue
        d = [(vs[n + 1][1] - vs[n][1], vs[n][0], vs[n + 1][0]) for n in range(len(vs) - 1)]
        big = max(d, key=lambda z: abs(z[0]))
        total = abs(i["v1"] - i["v0"])
        rows.append(dict(i, complete=done, series=vs, jump=big[0], ta=big[1], tb=big[2], width=big[2] - big[1],
                         share=abs(big[0]) / total if total else None))
    return rows


def main():
    stored = json.load(open(f"{CRIT}/dense_intervals.json"))
    now = {(i["tag"], i["k"], i["t0"], i["t1"], i["kind"]) for i in intervals()}
    coarse = {(p["tag"], p["t"]): (p["pd"], 16) for p in points()}
    dense = dense_v()
    rows = critical_rows(stored, coarse, dense)
    pending = {(j["tag"], j["t"]) for j in json.load(open(f"{DATA}/jobs_dense2_SteamInMicrowave.json"))} - set(dense)
    print(f"dense2 SteamInMicrowave points not yet in: {sorted(pending)}")
    for task in ["SteamInMicrowave", "PackIdenticalLunches"]:
        g = [r for r in rows if r["task"] == task]
        sel = [r for r in g if r["complete"] and abs(r["jump"]) >= 0.4]
        sh = [r["share"] for r in g if r["complete"] and r["share"] is not None]
        print(f"{task}: {len(g)} intervals, complete {sum(r['complete'] for r in g)}, |jump| >= 0.4: {len(sel)}; median share {np.median(sh):.2f}; "
              f"share > 1 in {sum(s > 1 for s in sh)}")
        n_coarse = n_dense = n_stale = 0
        for r in sel:
            key = (r["tag"], r["ta"])
            src = "coarse (continuations 0 and 1 reuse the seeds)" if key in coarse else ("dense (continuation 0 reuses the seeds)" if key in dense else "new")
            stale = (r["tag"], r["k"], r["t0"], r["t1"], r["kind"]) not in now
            n_coarse += key in coarse; n_dense += key in dense and key not in coarse; n_stale += stale
            print(f"   {r['tag']:31s} S{r['k']} [{r['t0']}, {r['t1']}] jump {r['jump']:+.2f} at [{r['ta']}, {r['tb']}) width {r['width']:3d}  "
                  f"ta is {src}{'  STALE INTERVAL' if stale else ''}")
        print(f"   -> {len(sel)} confirmation points: at a coarse point {n_coarse}, at a dense point {n_dense}, from stale intervals {n_stale}")


if __name__ == "__main__":
    main()
