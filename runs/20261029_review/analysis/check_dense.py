"""Checks of make_dense_jobs.py / dense_analysis.py on the real files (read-only; nothing in the project is written).
1. dense_intervals.json (written 2026-10-28 17:47) vs the intervals the current coarse rows give (make_dense_jobs.intervals()).
2. dense_analysis.main()'s per-interval logic, re-run with diagnostics:
   - the 'complete' flag (len(vs) >= len(ts)) vs every grid point present;
   - share > 1, and v0/v1 stored in the interval vs the current coarse values at its ends;
   - the hidden tags of the dense rows in each series (v3 grid + v4 refinement);
   - dense rows whose k or L differ from the interval's.
usage: python -I check_dense.py"""
import collections, glob, json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
CRIT = f"{R}/20261028_critical"
sys.path.insert(0, CRIT)
import numpy as np  # noqa: E402
from effect_by_event import RUNS, points  # noqa: E402
from make_dense_jobs import intervals  # noqa: E402


def dense_rows():
    out = {}
    for task in RUNS:
        for d in (f"run_dense_{task}", f"run_dense2_{task}"):
            for p in glob.glob(os.path.join(DATA, d, "results_w*.jsonl")):
                for line in open(p):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" not in r:
                        y = [x["done"] for c in r["cands"] for x in c["rollouts"]]
                        out.setdefault((r["tag"], r["t"]), []).append(dict(v=float(np.mean(y)), n=len(y), k=r["k"], L=r["L"], hidden=r.get("hidden"), dir=d))
    return out


def main():
    stored = json.load(open(f"{CRIT}/dense_intervals.json"))
    now = intervals()
    key = lambda i: (i["tag"], i["k"], i["t0"], i["t1"], i["kind"])
    ks, kn = {key(i): i for i in stored}, {key(i): i for i in now}
    print(f"1. intervals: stored {len(ks)}, from the current rows {len(kn)}")
    for k in sorted(set(ks) - set(kn)):
        print(f"   stale (no longer an interval): {k} stored v0 {ks[k]['v0']:.2f} v1 {ks[k]['v1']:.2f}")
    for k in sorted(set(kn) - set(ks)):
        print(f"   missing (an interval now, never densified): {k} v0 {kn[k]['v0']:.2f} v1 {kn[k]['v1']:.2f}")
    chg = [(k, ks[k]["v0"], kn[k]["v0"], ks[k]["v1"], kn[k]["v1"]) for k in set(ks) & set(kn) if (ks[k]["v0"], ks[k]["v1"]) != (kn[k]["v0"], kn[k]["v1"])]
    print(f"   same interval, changed end values: {len(chg)} {chg[:5]}")

    coarse = {(p["tag"], p["t"]): (p["pd"], 16) for p in points()}
    dense = dense_rows()
    dup = {k: v for k, v in dense.items() if len(v) > 1}
    print(f"2. dense (tag, t) with rows in two dirs: {len(dup)}")
    step = {"KettleBoiling": 16, "SteamInMicrowave": 32, "PackIdenticalLunches": 32}
    n_flag_wrong = n_share_gt1 = 0
    for i in stored:
        ts = list(range(i["t0"], i["t1"], step[i["task"]])) + [i["t1"]]
        have, extra, tags, bad_k = [], [], collections.Counter(), []
        for t in ts:
            if t == i["t1"] and i["kind"] == "rise_to_done":
                have.append(t)
            elif (i["tag"], t) in coarse and t in (i["t0"], i["t1"]):
                have.append(t)
            elif (i["tag"], t) in dense:
                have.append(t); tags[dense[(i["tag"], t)][-1]["hidden"]] += 1
        for t in range(i["t0"] + 16, i["t1"], 16):
            if t not in ts and (i["tag"], t) in dense:
                extra.append(t); tags[dense[(i["tag"], t)][-1]["hidden"]] += 1
        for t in ts[1:-1] + extra:
            for r in dense.get((i["tag"], t), []):
                if r["k"] != i["k"]:
                    bad_k.append((t, r["k"]))
        vs_len = len(have) + len(extra)
        flag = vs_len == len(ts) or vs_len > len(ts)
        truly = len(have) == len(ts)
        if flag != truly:
            n_flag_wrong += 1
            print(f"   'complete' wrong: {i['tag']} S{i['k']} [{i['t0']}, {i['t1']}] grid {len(ts)} present {len(have)} + refinement {len(extra)}; "
                  f"missing {[t for t in ts if t not in have]}")
        v_end0 = coarse.get((i["tag"], i["t0"]), (None,))[0]
        v_end1 = 1.0 if i["kind"] == "rise_to_done" else coarse.get((i["tag"], i["t1"]), (None,))[0]
        if (v_end0, v_end1) != (i["v0"], i["v1"]):
            print(f"   stored ends differ from the current coarse rows: {i['tag']} S{i['k']} [{i['t0']}, {i['t1']}] stored {i['v0']:.2f}->{i['v1']:.2f}, "
                  f"now {v_end0}->{v_end1}")
        if len(set(tags)) > 1:
            print(f"   mixed code versions in one series: {i['tag']} S{i['k']} [{i['t0']}, {i['t1']}] {dict(tags)}")
        if bad_k:
            print(f"   dense rows with another stage: {i['tag']} {bad_k}")
    print(f"   'complete' flag differs from 'every grid point present' in {n_flag_wrong} intervals")
    cc = json.load(open(f"{CRIT}/critical_chunks.json"))
    sh = [(r["tag"], r["k"], r["t0"], r["t1"], round(r["share"], 2)) for r in cc if r["share"] is not None and r["share"] > 1.0001]
    print(f"   critical_chunks.json rows with share > 1: {len(sh)} {sh[:8]}")
    nonmono = 0
    for r in cc:
        v = [x[1] for x in r["series"]]
        d = np.diff(v)
        if len(d) and (d.max() > 0.2 and d.min() < -0.2):
            nonmono += 1
    print(f"   series that go both up and down by > 0.2 between consecutive points: {nonmono} of {len(cc)}")


if __name__ == "__main__":
    main()
