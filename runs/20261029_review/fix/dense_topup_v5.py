"""Dense V points still missing in the intervals of the re-collected (settled-v5) coarse V curves.
make_dense_jobs.intervals() on the coarse rows in use; every chunk boundary of an interval (16 steps for KettleBoiling, 32
for the others, as make_dense_jobs) that has no row in the coarse, dense or dense2 data becomes a job (N = 8, M = 1, the
template job of its episode and stage), written to run_dense_<task>_fix5. Unlike make_dense_jobs.py it leaves
jobs_dense_<task>.json and dense_intervals.json alone.
usage: python -I dense_topup_v5.py jobs_out.json"""
import json, os, sys
C = "/home/michaellee/mclee/robocasa/runs/20261028_critical"
sys.path.insert(0, C)
import data  # noqa: E402
from make_dense_jobs import intervals, STEP  # noqa: E402

base = {t: {(j["tag"], j["t"]): j for j in json.load(open(os.path.join(data.DATA, f)))} for t, f in data.JOBS.items()}
have = {t: set(data.rows(t, "coarse")) | set(data.rows(t, "dense")) | set(data.rows(t, "dense2")) for t in data.TASKS}
jobs = []
for i in intervals():
    task = i["task"]
    tmpl = next(j for (tag, _), j in base[task].items() if tag == i["tag"] and j["k"] == i["k"])
    for t in range(i["t0"] + STEP[task], i["t1"], STEP[task]):
        if (i["tag"], t) not in have[task]:
            jobs.append(dict(tmpl, t=t, N=8, M=1, salt="", out=f"{data.DATA}/run_dense_{task}_fix5", replaces="(new dense point)"))
jobs = list({(j["out"], j["tag"], j["t"]): j for j in jobs}.values())
for t in data.TASKS:
    print(t, sorted({(j["tag"].split("_")[1], j["k"]) for j in jobs if j["task"] == t}), sum(j["task"] == t for j in jobs), "points")
json.dump(jobs, open(sys.argv[1], "w"), indent=1)
print(len(jobs), "jobs ->", sys.argv[1])
