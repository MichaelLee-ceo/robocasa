"""Offline checks on the result rows (no sim):
1. k_replay (stage rebuilt by collect.replay) vs the logged stage k;
2. obs_fp (fingerprint of the first policy request) identical across all rollouts of a point;
3. microwave._turned_on in py_state vs the main-run trace's fixture state;
4. outcome vs rollout order r = i*M + m inside a point (the observation-sampling phase advances by one physics substep
   per rollout, see report): within-point centred regression of done on r, and done by m.
usage: python -I rows_check.py -> out/rows_check.json"""
import glob, gzip, json, os, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_scan import rows  # noqa: E402

R = "/home/michaellee/mclee/robocasa/runs"
D = f"{R}/20261026_critic_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DIRS = ["run_v2/s0", "run_v2/s1", "run_v2/s2", "run_v2/s3", "run_v2/s4", "run_dense_KettleBoiling", "run_confirm_KettleBoiling",
        "run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix3", "run_dense_SteamInMicrowave", "run_dense2_SteamInMicrowave",
        "run_PackIdenticalLunches", "run_PackIdenticalLunches_fix", "run_PackIdenticalLunches_fix3",
        "run_dense_PackIdenticalLunches", "run_dense2_PackIdenticalLunches"]


def main():
    res = dict(k_mismatch=[], fp_mismatch=[], turned_on=[], order={})
    hist = {}
    for p in glob.glob(f"{R}/20261025_seen480/run/results_w*.jsonl"):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "error" not in r and r.get("tag", "").endswith("_single"):
                hist[r["tag"]] = r
    traces = {}
    for d in DIRS:
        per_r = defaultdict(list)  # (centred r, centred done)
        by_m = defaultdict(list)
        by_rank = defaultdict(list)
        n_pts = 0
        for r in rows(d):
            n_pts += 1
            if r.get("k_replay") != r["k"]:
                res["k_mismatch"].append(dict(dir=d, tag=r["tag"], t=r["t"], k=r["k"], k_replay=r.get("k_replay")))
            fps = {ro["obs_fp"] for c in r["cands"] for ro in c["rollouts"]}
            if len(fps) > 1:
                res["fp_mismatch"].append(dict(dir=d, tag=r["tag"], t=r["t"], n=len(fps)))
            mw = r.get("py_state", {}).get("microwave")
            if mw is not None and "_turned_on" in mw:
                task = r["task"]
                if r["tag"] not in traces:
                    f = f"{R}/20261025_seen480/analysis/{task}/{r['tag']}_trace.json.gz"
                    traces[r["tag"]] = json.load(gzip.open(f)) if os.path.exists(f) else None
                res["turned_on"].append(dict(dir=d, tag=r["tag"], t=r["t"], k=r["k"], py=mw["_turned_on"]))
            M = r["M"]
            dn = [(ci * M + ro["m"], ro["m"], float(ro["done"])) for ci, c in enumerate(r["cands"]) for ro in c["rollouts"]]
            ds = np.array([x[2] for x in dn])
            if ds.min() == ds.max():
                continue  # no within-point information
            rs = np.array([x[0] for x in dn], float)
            for (rr, m, v) in dn:
                per_r[d].append((rr - rs.mean(), v - ds.mean()))
                by_m[m].append(v - ds.mean())
                by_rank[rr].append(v - ds.mean())
        x = np.array(per_r[d]) if per_r[d] else np.zeros((0, 2))
        if len(x):
            slope = float((x[:, 0] * x[:, 1]).sum() / (x[:, 0] ** 2).sum())
            # bootstrap over points is better; quick SE from residuals
            resid = x[:, 1] - slope * x[:, 0]
            se = float(np.sqrt((resid ** 2).sum() / max(len(x) - 1, 1) / (x[:, 0] ** 2).sum()))
        else:
            slope = se = None
        res["order"][d] = dict(points=n_pts, mixed_rollouts=len(x), slope_per_rollout=slope, se=se,
                               by_m={m: round(float(np.mean(v)), 4) for m, v in sorted(by_m.items())},
                               by_r={rr: round(float(np.mean(v)), 3) for rr, v in sorted(by_rank.items())})
        print(d, res["order"][d]["points"], "slope", slope, "+-", se, res["order"][d]["by_m"], flush=True)
    print("k mismatches:", len(res["k_mismatch"]), res["k_mismatch"][:10])
    print("obs_fp mismatches:", len(res["fp_mismatch"]), res["fp_mismatch"][:5])
    print("turned_on rows:", Counter((x["dir"], x["k"], x["py"]) for x in res["turned_on"]))
    json.dump(res, open(f"{OUT}/rows_check.json", "w"), default=str)


if __name__ == "__main__":
    main()
