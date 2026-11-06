"""Score the recovery loop (loop/<arm>/*.json, one per run; errors from loop/results*.jsonl): per arm, how often the failed stage gets done ("fixed") and the task
succeeds, by task and by failed stage; which intervention was the last one before the fix; paired per failure point.
usage: python -I loop_score.py"""
import collections, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
import glob
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "loop", "*", "*_p*.json"))]  # one file per finished run
err = [json.loads(l) for f in glob.glob(os.path.join(HERE, "loop", "results*.jsonl")) for l in open(f) if '"error"' in l]
rows = [r for r in rows if "error" not in r]
ORDER = ["control", "hold", "back", "full", "reset", "retrace"]
arms = sorted({r["arm"] for r in rows}, key=lambda a: ORDER.index(a) if a in ORDER else 9)
print(f"{len(rows)} runs, {len(err)} errors")


def rate(R, key):
    return f"{sum(r[key] for r in R)}/{len(R)}" if R else "-"


print("\n| task | failed stage | points | " + " | ".join(f"{a}: fixed / success" for a in arms) + " |")
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["task"], r["stage"])].append(r)
for (task, st), R in sorted(groups.items()):
    pts = len({r["id"] for r in R})
    cells = [f"{rate([r for r in R if r['arm'] == a], 'fixed')} / {rate([r for r in R if r['arm'] == a], 'success')}" for a in arms]
    print(f"| {task} | {st} | {pts} | " + " | ".join(cells) + " |")
for task in sorted({r["task"] for r in rows}):
    R = [r for r in rows if r["task"] == task]
    cells = [f"{rate([r for r in R if r['arm'] == a], 'fixed')} / {rate([r for r in R if r['arm'] == a], 'success')}" for a in arms]
    print(f"| **{task}** | all | {len({r['id'] for r in R})} | " + " | ".join(cells) + " |")
cells = [f"{rate([r for r in rows if r['arm'] == a], 'fixed')} / {rate([r for r in rows if r['arm'] == a], 'success')}" for a in arms]
print(f"| **all** | | {len({r['id'] for r in rows})} | " + " | ".join(cells) + " |")

print("\nlast intervention before the failed stage was fixed (full):")
c = collections.Counter()
for r in rows:
    if r["arm"] != "full" or not r["fixed"]:
        continue
    iv = [x for x in r["log"] if x["event"] == "intervene" and x["stage"] == r["stage"]
          and x["chunk"] - r["start_chunk"] <= (r["fixed_after"] or 0)]
    c[(iv[-1]["attempt"], iv[-1]["kind"], iv[-1]["back"], iv[-1]["instruction"]) if iv else "none"] += 1
for k, v in sorted(c.items(), key=lambda kv: str(kv[0])):
    print(f"  {k}: {v}")

print("\nretrace / reset: the failed stage fixed after the first one, at all, and how many per run:")
R = [r for r in rows if r["arm"] in ("reset", "retrace")]
if R:
    first = [r for r in R if r["fixed"] and sum(1 for x in r["log"] if x["event"] == "intervene"
                                                 and x["stage"] == r["stage"]) == 1]
    print(f"  fixed after one reset {len(first)}/{len(R)}, fixed at all {sum(r['fixed'] for r in R)}/{len(R)};",
          "resets per run:", dict(collections.Counter(r["interventions"] for r in R)))

print("\npaired per failure point vs control (rate over the seeds; sign test):")
pts = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    pts[r["id"]][r["arm"]].append(r)
for arm in arms[1:]:
    for key in ("fixed", "success"):
        for task in sorted({r["task"] for r in rows}) + [None]:
            b = w = 0
            for pid, d in pts.items():
                if (task and not pid.startswith(task)) or not d["control"] or not d[arm]:
                    continue
                a = sum(r[key] for r in d["control"]) / len(d["control"])
                c = sum(r[key] for r in d[arm]) / len(d[arm])
                b += c > a; w += c < a
            n = b + w
            p = min(1.0, 2 * sum(math.comb(n, k) for k in range(max(b, w), n + 1)) / 2 ** n) if n else 1.0
            print(f"  {arm:6s} {key:8s} {task or 'all':22s} better {b:2d}, worse {w:2d}, p {p:.3f}")
