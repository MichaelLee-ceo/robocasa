import json, glob, gzip, os, collections
R = "/home/michaellee/mclee/robocasa/runs/20261025_seen480"
rows = {}
for p in glob.glob(R + "/run/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r: rows[r["tag"]] = r
bad = collections.Counter(); n = collections.Counter(); ex = []
for tag, r in sorted(rows.items()):
    f = f"{R}/analysis/{r['task']}/{tag}_trace.json.gz"
    tr = json.load(gzip.open(f))
    st = tr["stage"]
    for t, k in r["history"]:
        n[r["task"]] += 1
        if not st[t][k]:
            bad[r["task"]] += 1; ex.append((tag, t, k, st[t]))
    # also: live final k vs replay-ordered progression at chunk boundaries
    k = 0; hist2 = []
    for t in range(16, len(st), 16) if True else []:
        while k < len(st[t]) and st[t][k]:
            hist2.append([t, k]); k += 1
    T = len(st) - 1
    if T % 16:
        while k < len(st[T]) and st[T][k]:
            hist2.append([T, k]); k += 1
    if hist2 != r["history"]:
        ex.append(("ordered-progression differs", tag, r["history"], hist2))
for t in sorted(n): print(t, "history entries", n[t], "not true in replay", bad[t])
for e in ex: print(e)
