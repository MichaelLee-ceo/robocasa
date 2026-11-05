import sys, json, glob, gzip, os
from collections import defaultdict
R = "/home/michaellee/mclee/robocasa/runs/20261025_seen480"
rs = {}
for p in glob.glob(f"{R}/run/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" in r: continue
        rs[r["tag"]] = r
per = defaultdict(lambda: dict(n=0, ours=0, live_mid_only=0, tr_any=0, tr_ce=0, tr_last=0, mism=0))
ex = []
for tag, r in sorted(rs.items()):
    t = r["task"]; d = per[t]; d["n"] += 1
    d["ours"] += r["success"]
    if r["success"] and r["reached"] < r["n_steps"]:
        d["live_mid_only"] += 1
    fp = f"{R}/analysis/{t}/{tag}_trace.json.gz"
    if not os.path.exists(fp):
        continue
    st = json.load(gzip.open(fp))["stage"]
    fin = [s[-1] for s in st]
    T = len(fin) - 1
    anyf = any(fin)
    ce = any(fin[f] for f in range(16, T + 1, 16)) or (T % 16 != 0 and fin[T])
    d["tr_any"] += anyf; d["tr_ce"] += ce; d["tr_last"] += fin[T]
    if anyf != r["success"]: d["mism"] += 1
    if anyf and not ce:
        frames = [f for f in range(T + 1) if fin[f]]
        ex.append((tag, r["steps"], T, frames[:3], frames[-3:], len(frames), r["reached"], r["n_steps"]))
tot = defaultdict(int)
print(f"{'task':24s} n ours live_midonly tr_any tr_chunkend tr_lastframe replay_mismatch")
for t, d in sorted(per.items()):
    print(f"{t:24s} {d['n']:2d} {d['ours']:3d} {d['live_mid_only']:3d} {d['tr_any']:3d} {d['tr_ce']:3d} {d['tr_last']:3d} {d['mism']:3d}")
    for k, v in d.items(): tot[k] += v
print("TOTAL", dict(tot))
for e in ex: print(e)
