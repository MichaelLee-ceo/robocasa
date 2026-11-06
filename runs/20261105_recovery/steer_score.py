"""Score the steerability probe (steer/probe/).

1. Action effect of the instruction, first chunk. The restored observation is identical and the policy seed is shared
   across instructions, so with the same seed any difference from the task sentence's chunk comes from the
   instruction alone. effect = mean over seeds |chunk(instruction, seed) - chunk(orig, seed)|, relative to the policy's
   own spread: mean |chunk(orig, seed a) - chunk(orig, seed b)| over seed pairs. About 0 = the instruction is ignored;
   about 1 = it changes the chunk as much as a new noise sample does.
2. Outcome within the probe's chunks: stage done (of the seeds), hand-to-target distance at the end, the best target
   value (drawer opening, lever; grasped for pick stages), base displacement.
3. With STEER_DIR=steer/v2: also the predicted-chunk ranking (steer_rank.py: toward_end) next to the outcome.
usage: python -I steer_score.py [ids...]   (env STEER_DIR, default steer)"""
import glob, gzip, json, os, pickle, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, os.environ.get("STEER_DIR", "steer"))
PROBE = os.path.join(DIR, "probe")
KEYS = ("action.end_effector_position", "action.end_effector_rotation", "action.gripper_close", "action.base_motion",
        "action.control_mode")


def vec(c):
    return np.concatenate([np.asarray(c[k], dtype=float).ravel() for k in KEYS])


def score(iid):
    rows, acts = [], {}
    for fn in glob.glob(os.path.join(PROBE, f"{iid}.jsonl")) + glob.glob(os.path.join(PROBE, f"{iid}_k*.jsonl")):
        rows += [json.loads(l) for l in open(fn)]
    for fn in glob.glob(os.path.join(PROBE, f"{iid}_actions*.pkl.gz")):
        acts.update(pickle.load(gzip.open(fn)))
    rank = {}
    if os.path.isfile(os.path.join(DIR, "rank", f"{iid}.json")):
        rank = {r["k"]: r for r in json.load(open(os.path.join(DIR, "rank", f"{iid}.json")))["rows"]}
    seeds = sorted({p for _, p in acts})
    orig_k = next(r["k"] for r in rows if r["family"] == "orig")
    o = {p: vec(acts[(orig_k, p)]) for p in seeds}
    noise = np.mean([np.linalg.norm(o[a] - o[b]) for a in seeds for b in seeds if a < b])
    out = []
    for k in sorted({r["k"] for r in rows}):
        R = [r for r in rows if r["k"] == k]
        eff = np.mean([np.linalg.norm(vec(acts[(k, p)]) - o[p]) for p in seeds]) / noise
        done = [next((i + 1 for i, c in enumerate(r["chunks"]) if c["done"]), None) for r in R]
        out.append(dict(k=k, family=R[0]["family"], text=R[0]["text"], effect=round(float(eff), 2),
                        done=sum(d is not None for d in done), n=len(R), done_at=done,
                        dist_start=R[0]["start"]["dist"], dist_end=round(float(np.mean([r["chunks"][-1]["dist"] for r in R])), 3),
                        dist_min=round(float(np.mean([min(c["dist"] for c in r["chunks"]) for r in R])), 3),
                        value_start=R[0]["start"]["value"],
                        value_best=round(float(np.mean([max(c["value"] for c in r["chunks"]) for r in R])), 3),
                        base_moved=round(float(np.mean([r["chunks"][-1]["base_moved"] for r in R])), 3),
                        toward=rank.get(k, {}).get("toward_end")))
    return out


if __name__ == "__main__":
    import re
    ids = sys.argv[1:] or sorted({re.sub(r"_k[0-9-]+$", "", os.path.basename(f)[:-6]) for f in glob.glob(os.path.join(PROBE, "*.jsonl"))})
    allrows = []
    for iid in ids:
        res = score(iid)
        allrows += [dict(id=iid, **r) for r in res]
        r0 = res[0]
        print(f"\n## {iid}  (start: dist {r0['dist_start']}, value {r0['value_start']})")
        print("family   effect done  dist_end dist_min value_best base  toward  text")
        for r in res:
            print(f"{r['family']:8s} {r['effect']:5.2f}  {r['done']}/{r['n']}   {r['dist_end']:6.3f}  {r['dist_min']:6.3f}   "
                  f"{r['value_best']:6.3f}  {r['base_moved']:5.2f}  {'' if r['toward'] is None else format(r['toward'], '+.3f'):6s}  {r['text'][:80]}")
    with open(os.path.join(DIR, "scored.jsonl"), "w") as f:
        for r in allrows:
            f.write(json.dumps(r) + "\n")
    fam = {}
    for r in allrows:
        fam.setdefault(r["family"], []).append(r)
    print("\nby family (all failure points): effect mean | stage done")
    for k, v in fam.items():
        print(f"  {k:8s} {np.mean([r['effect'] for r in v]):.2f} | {sum(r['done'] for r in v)}/{sum(r['n'] for r in v)}")
