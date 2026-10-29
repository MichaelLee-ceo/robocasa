"""Refinement points (dense2) for the 2-chunk jumps of the relocated critical chunks: dense_analysis.analyse()'s refine
list minus the points already collected, as settled-v5 jobs into run_dense2_<task>_fix5 (jobs_dense2_<task>.json is left
alone). usage: python -I make_refine_v5.py jobs_out.json"""
import json, sys
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/20261028_critical")
import data  # noqa: E402
from dense_analysis import analyse  # noqa: E402

_, refine, _ = analyse()
jobs = []
for task in data.TASKS:
    have = set(data.rows(task, "dense")) | set(data.rows(task, "dense2"))
    for j in refine[task]:
        if (j["tag"], j["t"]) not in have:
            jobs.append(dict(j, salt="", out=f"{data.DATA}/run_dense2_{task}_fix5", replaces="(new refinement point)"))
jobs = list({(j["out"], j["tag"], j["t"]): j for j in jobs}.values())
for t in data.TASKS:
    print(t, sum(j["task"] == t for j in jobs), "refinement points")
json.dump(jobs, open(sys.argv[1], "w"), indent=1)
print(len(jobs), "jobs ->", sys.argv[1])
