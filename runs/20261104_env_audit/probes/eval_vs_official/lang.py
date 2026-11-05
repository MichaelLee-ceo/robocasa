import json, glob, sys, collections
import numpy as np
sys.path.insert(0, "/home/michaellee/mclee/robocasa/runs/scripts")
from paths import local
R = "/home/michaellee/mclee/robocasa/runs/20261025_seen480/run"
rows = [json.loads(l) for p in glob.glob(R + "/results_w*.jsonl") for l in open(p)]
rows = [r for r in rows if "error" not in r]
print(len(rows), "rows")
bad = collections.Counter(); n = collections.Counter(); ex = {}
succ = collections.Counter()
for r in rows:
    z = np.load(local(r["scene"]), allow_pickle=True)
    lang = str(z["lang"]); meta = json.loads(str(z["ep_meta"]))
    n[r["task"]] += 1; succ[r["task"]] += r["success"]
    if r["task_instr"] != lang or meta["lang"] != lang:
        bad[r["task"]] += 1; ex.setdefault(r["task"], (r["tag"], lang, r["task_instr"]))
for t in sorted(n): print(t, n[t], "succ", succ[t], "lang mismatch", bad[t])
for t, e in ex.items(): print(e)
