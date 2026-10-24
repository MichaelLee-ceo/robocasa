"""Scene-identity check from videos (no new rollouts).
(1) every fresh-scene episode: first frame (agentview half) vs the baseline_seed episode of the same task/seed.
(2) every restore (reset_seed_*): its first frame vs the source run's frame at the saved time t."""
import json, glob, os, sys, collections
import numpy as np, av
SD = os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa") + "/runs/20261012_atomic_seeded")
R = {}
for p in glob.glob(f"{SD}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            R[r["tag"]] = r
_cache = {}
def frames(tag, task):
    k = (task, tag)
    if k not in _cache:
        c = av.open(f"{SD}/{task}/{tag}.mp4")
        _cache[k] = [f.to_ndarray(format="rgb24")[:, :256].astype(np.float32) for f in c.decode(video=0)]
        c.close()
    return _cache[k]
def mad(a, b):
    return float(np.abs(a - b).mean())
out = dict(first=[], restore=[])
for r in R.values():
    if r["condition"].startswith("reset_seed") or "seed" not in r or r.get("demo") is not None:
        continue
    ref = f"{r['task']}_s{r['seed']}_baseline_seed"
    if r["tag"] == ref or ref not in R:
        continue
    try:
        out["first"].append(dict(tag=r["tag"], cond=r["condition"], d=mad(frames(r["tag"], r["task"])[0], frames(ref, r["task"])[0])))
    except Exception as e:
        out["first"].append(dict(tag=r["tag"], cond=r["condition"], d=None, err=repr(e)))
    _cache.clear()
for r in R.values():
    if not r["condition"].startswith("reset_seed"):
        continue
    src = r["src_tag"]
    try:
        F = frames(src, r["task"]); i = r["start_t"] // 4 - 1
        f0 = frames(r["tag"], r["task"])[0]
        out["restore"].append(dict(tag=r["tag"], cond=r["condition"], d=mad(f0, F[i]), d_next=mad(f0, F[min(i + 1, len(F) - 1)]),
                                   d_other=mad(f0, frames(f"{r['task']}_s{(r['seed'] + 1) % 20}_baseline_seed_save", r["task"])[0])))
    except Exception as e:
        out["restore"].append(dict(tag=r["tag"], cond=r["condition"], d=None, err=repr(e)))
    _cache.clear()
json.dump(out, open(f"{SD}/scene_check.json", "w"), indent=1)
for k, v in out.items():
    ds = sorted([x["d"] for x in v if x["d"] is not None])
    print(k, len(v), "errors", sum(x["d"] is None for x in v), "quantiles", [round(ds[int(q * (len(ds) - 1))], 1) for q in (0, .5, .9, .99, 1)] if ds else None)
    for x in sorted(v, key=lambda x: -(x["d"] or 1e9))[:8]:
        print("   ", x)
