import json, glob, gzip, os, collections
R = "/home/michaellee/mclee/robocasa/runs/20261025_seen480"
rows = {}
for p in glob.glob(R + "/run/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r: rows[r["tag"]] = r
stat = collections.defaultdict(collections.Counter)
for tag, r in sorted(rows.items()):
    f = f"{R}/analysis/{r['task']}/{tag}_trace.json.gz"
    st = stat[r["task"]]; st["n"] += 1; st["ours"] += r["success"]
    if not os.path.exists(f):
        st["no_trace"] += 1; continue
    tr = json.load(gzip.open(f))
    fin = [s[-1] for s in tr["stage"]]
    T = len(fin) - 1
    assert T == r["steps"], (tag, T, r["steps"])
    first = next((i for i, v in enumerate(fin) if v), None)
    ends = list(range(16, T + 1, 16)) + ([T] if T % 16 else [])
    ce = [i for i in ends if fin[i]]
    st["trace_any"] += first is not None
    st["trace_chunkend"] += bool(ce)
    if r["success"] != (first is not None):
        print("MISMATCH result vs trace", tag, r["success"], first, T)
    if first is not None and not ce:
        run = fin[first:T + 1]
        print("midchunk_only", tag, "first", first, "T", T, "true_frames", sum(run), "of", T + 1 - first,
              "true_idx", [i for i in range(first, T + 1) if fin[i]])
tot = collections.Counter()
for t in sorted(stat):
    print(t, dict(stat[t])); tot.update(stat[t])
print("TOTAL", dict(tot))
