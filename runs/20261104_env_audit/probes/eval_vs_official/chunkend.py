import json, glob, gzip, os, collections
R = "/home/michaellee/mclee/robocasa/runs/20261025_seen480"
rows = {}
for p in glob.glob(R + "/run/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r: rows[r["tag"]] = r
stat = collections.defaultdict(lambda: collections.Counter())
details = []
for tag, r in sorted(rows.items()):
    f = f"{R}/analysis/{r['task']}/{tag}_trace.json.gz"
    if not os.path.exists(f):
        stat[r["task"]]["no_trace"] += 1; continue
    tr = json.load(gzip.open(f))
    fin = [s[-1] for s in tr["stage"]]
    T = len(fin) - 1
    first = next((i for i, v in enumerate(fin) if v), None)
    chunk_end_true = any(fin[i] for i in range(16, T + 1, 16)) or (fin[T])
    st = stat[r["task"]]
    st["n"] += 1
    st["succ_result"] += r["success"]
    st["succ_trace_any"] += first is not None
    st["succ_chunkend"] += chunk_end_true
    if r["success"] != (first is not None):
        details.append(("result_vs_trace", tag, r["success"], first, T, r["steps"]))
    if (first is not None) and not chunk_end_true:
        details.append(("midchunk_only", tag, first, T, fin[first:T+1].count(True), T + 1 - first))
for t in sorted(stat): print(t, dict(stat[t]))
for d in details: print(d)
