"""Failure points from phase 0: for every failed episode, the stage it ends in and the chunk where an oracle supervisor
would step in.
  undone   the stage was done earlier and later undone (e.g. the vegetable fell out of the bowl): the first chunk end
           of UNDONE_N in a row where it is not done (single-chunk flickers are ignored)
  stalled  the stage has been the current one for longer than STALL_Q of the completions of that stage in the
           successful and failed episodes (phase0_summary.py): stage start + that many chunks
  late     the stall point is after the episode's end (no time left to act); kept for the count only
Output: phase0/points.json  [{id, task, seed, stage, kind, chunk, stage_start, chunks_left, events}]
usage: python -I phase0_points.py"""
import collections, gzip, json, math, os, pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STALL_Q, UNDONE_N = 90, 3

rows = [json.loads(l) for l in open(os.path.join(HERE, "phase0", "plain.jsonl")) if l.strip()]
rows = [r for r in rows if "error" not in r]
dur = collections.defaultdict(list)
for r in rows:
    fd = {int(k): v for k, v in r["first_done"].items()}
    for k in range(r["n_stages"]):
        if k in fd and (k == 0 or k - 1 in fd):
            dur[(r["task"], k)].append((fd[k] - (fd[k - 1] if k else 0)) / 16)
limit = {k: math.ceil(np.percentile(v, STALL_Q)) for k, v in dur.items()}

points = []
for r in sorted(rows, key=lambda r: (r["task"], r["seed"])):
    if r["success"]:
        continue
    rec = pickle.load(gzip.open(os.path.join(HERE, "phase0", "records", f"{r['task']}_s{r['seed']}.pkl.gz")))
    tr = rec["tracks"]  # tr[i] = after i chunks
    s = tr[-1]["stage"]
    n = len(rec["chunks"])
    start = next(i for i, t in enumerate(tr) if t["stage"] >= s)  # first time all stages before s are done
    ever = next((i for i, t in enumerate(tr) if t["done"][s]), None)
    evs = [e for xs in rec["events"] for e in xs if e["stage"] == s]
    if ever is not None:
        first = next(i for i in range(ever, len(tr)) if not tr[i]["done"][s])
        kind, chunk = "undone", next((i for i in range(ever, len(tr) - UNDONE_N + 1)
                                      if not any(tr[j]["done"][s] for j in range(i, i + UNDONE_N))), first)
    else:
        kind, chunk = "stalled", start + limit.get((r["task"], s), 40)
        if chunk >= n:
            kind = "late"
    points.append(dict(id=f"{r['task']}_s{r['seed']}", task=r["task"], seed=r["seed"], stage=s, kind=kind,
                       chunk=min(chunk, n) - 1, stage_start=start, chunks_left=n - min(chunk, n),
                       events=[(e["type"], e["chunk"]) for e in evs][:4]))
json.dump(points, open(os.path.join(HERE, "phase0", "points.json"), "w"), indent=1)
print("stall limits (chunks):", {f"{t}/{k}": v for (t, k), v in sorted(limit.items())})
for task in sorted({p["task"] for p in points}):
    P = [p for p in points if p["task"] == task]
    print(f"\n## {task}: {len(P)} failed;", dict(collections.Counter((p['stage'], p['kind']) for p in P)))
    for p in P:
        print(f"  s{p['seed']:<2} stage {p['stage']} {p['kind']:7s} at chunk {p['chunk']:3d} (stage began {p['stage_start']:3d}, "
              f"{p['chunks_left']:3d} chunks left) {p['events']}")
