"""Locate the critical chunk in every interval of make_dense_jobs.py: the chunk where V jumps the most.

The intervals are recomputed from the coarse rows in use (make_dense_jobs.intervals()), not read from
dense_intervals.json: an interval whose ends no longer qualify is dropped, and one that newly qualifies is listed as
incomplete until its dense points are collected. Both are printed against dense_intervals.json, the list the dense jobs
were made from. V at the interval's ends comes from the coarse points (16 rollouts each; V = 1 at the main run's finish of
the stage for "rise_to_done" intervals). V inside comes from the dense points (8 plain rollouts each; data.py "dense",
refinement points included). Per interval:
  - the V series along the main run;
  - complete: every point of its grid is in (the ends and the dense points of make_dense_jobs; refinement points do not
    count);
  - the largest jump between consecutive points (the critical chunk lies between them: 1 chunk for KettleBoiling, 2
    for the others until refined), with its standard error from the binomial counts, sqrt(sum V(1 - V) / n) over its
    two points (0 at an exact finish), and noise = |jump| <= 2 SE. The SE is the plug-in one, small near V = 0 or 1;
  - share: |jump| / |V(t1) - V(t0)|, the series' own ends (1 = the outcome is decided in one step, low = it drifts; above
    1 when the series is not monotone);
  - the main run's events the jump's chunks can cause, at steps (ta, tb].
Writes critical_chunks.json (to $CRIT_OUT). With --refine, also writes ../20261026_critic_data/jobs/refine_<task>.json:
a point in the middle of every 2-chunk jump, to bring it down to one chunk. usage: python -I dense_analysis.py [--refine]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
from data import DATA, HERE, JOBS, OUT, TASKS  # noqa: E402
from effect_by_event import points  # noqa: E402
from make_dense_jobs import STEP, intervals  # noqa: E402

REFINE = "--refine" in sys.argv


def dense_v():
    """{(tag, t): (V, rollouts)} at the dense points in use; a refinement point (dense2) replaces a grid one"""
    out = {}
    for task in TASKS:
        for ds in ("dense",):
            for key, r in data.rows(task, ds).items():
                y = data.outcomes(r)
                out[key] = (float(y.mean()), y.size)
    return out


def key(i):
    return i["tag"], i["k"], i["t0"], i["t1"], i["kind"]


def analyse():
    """(rows of critical_chunks.json, refinement jobs per task, stored intervals no longer qualifying)"""
    iv = intervals()
    stored = json.load(open(os.path.join(HERE, "dense_intervals.json")))
    dropped = [i for i in stored if key(i) not in {key(j) for j in iv}]
    ev = {r["tag"]: r for r in map(json.loads, open(os.path.join(HERE, "events.jsonl")))}
    coarse = {(p["tag"], p["t"]): (p["pd"], 16) for p in points()}
    dense = dense_v()
    old = {key(i) for i in stored}
    rows, refine = [], {t: [] for t in TASKS}
    for i in iv:
        ts = list(range(i["t0"], i["t1"], STEP[i["task"]])) + [i["t1"]]
        vs = {}
        for t in ts:
            if t == i["t1"] and i["kind"] == "rise_to_done":
                vs[t] = (1.0, 0)
            elif (i["tag"], t) in coarse and t in (i["t0"], i["t1"]):
                vs[t] = coarse[(i["tag"], t)]
            elif (i["tag"], t) in dense:
                vs[t] = dense[(i["tag"], t)]
        complete = len(vs) == len(ts)
        for t in range(i["t0"] + 16, i["t1"], 16):  # refinement points, if collected
            if t not in vs and (i["tag"], t) in dense:
                vs[t] = dense[(i["tag"], t)]
        vs = sorted((t, v, n) for t, (v, n) in vs.items())
        if len(vs) < 2:
            continue
        se = lambda v, n: v * (1 - v) / n if n else 0.0
        d = [(b[1] - a[1], a[0], b[0], float(np.sqrt(se(*a[1:]) + se(*b[1:])))) for a, b in zip(vs[:-1], vs[1:])]
        big = max(d, key=lambda z: abs(z[0]))
        total = abs(vs[-1][1] - vs[0][1]) if (vs[0][0], vs[-1][0]) == (i["t0"], i["t1"]) else 0
        evs = [f"{e['t']}:{e['type']}" for e in ev[i["tag"]]["events"] if big[1] < e["t"] <= big[2] and e["type"] not in ("reach",)]
        rows.append(dict(i, complete=complete, new=key(i) not in old, series=vs, jump=big[0], ta=big[1], tb=big[2], width=big[2] - big[1],
                         se=big[3], noise=bool(abs(big[0]) <= 2 * big[3]), share=abs(big[0]) / total if total else None, events=evs))
        if complete and big[2] - big[1] > 16:
            tmpl = next(j for j in json.load(open(os.path.join(DATA, JOBS[i["task"]]))) if j["tag"] == i["tag"] and j["k"] == i["k"])
            for t in range(big[1] + 16, big[2], 16):
                refine[i["task"]].append(dict(tmpl, t=t, N=8, M=1))
    return rows, refine, dropped


def main():
    rows, refine, dropped = analyse()
    json.dump(rows, open(os.path.join(OUT, "critical_chunks.json"), "w"), indent=0)
    for task in TASKS:
        a = [r for r in rows if r["task"] == task]
        g = [r for r in a if r["complete"]]
        sh = [r["share"] for r in g if r["share"] is not None]
        print(f"{task:21s} intervals with every dense point in: {len(g)} / {len(a)}; largest single jump >= 0.4 in "
              f"{sum(abs(r['jump']) >= 0.4 for r in g)}, within 2 SE of noise in {sum(r['noise'] for r in g)}; median share of the "
              f"interval's change in one step {np.median(sh) if sh else float('nan'):.2f}")
        for i in [i for i in dropped if i["task"] == task]:
            print(f"    dropped, its ends no longer qualify: {i['tag']} S{i['k']} [{i['t0']}, {i['t1']}] {i['kind']} (stored V {i['v0']:.2f} -> {i['v1']:.2f})")
        for r in [r for r in a if not r["complete"]]:
            print(f"    incomplete{' (new interval, not densified)' if r['new'] else ''}: {r['tag']} S{r['k']} [{r['t0']}, {r['t1']}] {r['kind']} "
                  f"V {r['v0']:.2f} -> {r['v1']:.2f}")
        for r in sorted(g, key=lambda r: -abs(r["jump"]))[:5]:
            ser = " ".join(f"{t}:{v:.2f}" for t, v, _ in r["series"])
            print(f"    {r['tag']} S{r['k']} ({r['ta']}, {r['tb']}] dV {r['jump']:+.2f} (SE {r['se']:.2f}{', noise' if r['noise'] else ''})  "
                  f"series {ser}  events {' '.join(r['events'][:6])}")
        if REFINE:
            js = list({(j["tag"], j["t"]): j for j in refine[task]}.values())
            json.dump(js, open(os.path.join(DATA, "jobs", f"refine_{task}.json"), "w"), indent=0)
            print(f"    refinement points written: {len(js)}")


if __name__ == "__main__":
    main()
