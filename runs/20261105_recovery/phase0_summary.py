"""Summarise phase 0 (phase0/plain.jsonl and records): success per task, where failed episodes stop (the first stage
not done at the end), and how long each stage takes when it is done (chunks from the previous stage's completion), so
that a "stalled stage" can be defined per task (longer than the slowest completions).
usage: python -I phase0_summary.py"""
import collections, gzip, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import numpy as np  # noqa: E402

rows = [json.loads(l) for l in open(os.path.join(HERE, "phase0", "plain.jsonl")) if l.strip()]
rows = [r for r in rows if "error" not in r]
by = collections.defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
for task, R in sorted(by.items()):
    n_st = R[0]["n_stages"]
    print(f"\n## {task}: success {sum(r['success'] for r in R)}/{len(R)}  ({n_st} stages)")
    stuck = collections.Counter(r["final_stage"] for r in R if not r["success"])
    print("  failed episodes end at stage:", dict(sorted(stuck.items())))
    durs = collections.defaultdict(list)  # chunks to finish stage k after stage k-1 (first completion times)
    for r in R:
        fd = {int(k): v for k, v in r["first_done"].items()}
        for k in range(n_st):
            if k in fd and (k == 0 or k - 1 in fd):
                durs[k].append((fd[k] - (fd[k - 1] if k else 0)) / 16)
    for k in range(n_st):
        d = durs.get(k, [])
        if d:
            print(f"  stage {k}: done in {len(d)} episodes; chunks median {np.median(d):.0f}, 90 % {np.percentile(d, 90):.0f}, max {max(d):.0f}")
        else:
            print(f"  stage {k}: never done")
    ev = collections.Counter(e["type"] for r in R for e in r.get("events", []))
    if ev:
        print("  oracle events:", dict(ev))
