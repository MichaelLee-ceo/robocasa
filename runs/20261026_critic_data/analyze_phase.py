"""Does the candidate chunk change the stage outcome, and where in the stage? (the coarse rows in use)

The rows are the task's coarse rows (../20261028_critical/data.py, data/<task>/coarse).
For each decision point, the two continuations of a candidate share its first chunk and differ only afterwards, so
the covariance of the two continuations' outcomes across candidates estimates the between-candidate variance of
P(stage done | candidate) without bias (zero if the candidate does not matter). Reported per position in the stage, with
95 % bootstrap intervals over decision points, plus the same/cross agreement and the gains of
../20261028_critical/effect_by_event.py: a one-rollout critic, cross-validated (the headline), a perfect best-of-8
picker computed per point, and the pooled 142 sqrt(mean var_b), which is the gain at a point of average variance and
overstates the mean gain.
usage: python -I analyze_phase.py <task> [main run dir]"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "20261028_critical"))
from effect_by_event import TASKS, points, summary as effect  # noqa: E402

task = sys.argv[1] if len(sys.argv) > 1 else None
if task not in TASKS:  # it took a glob of results files before 2026-10-29
    sys.exit(__doc__)
MAIN = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "20261025_seen480", "run")
main = {r["tag"]: r for p in glob.glob(f"{MAIN}/results_w*.jsonl") for r in map(json.loads, open(p))}
pts = [p for p in points() if p["task"] == task]
for p in pts:
    hist = [t for t, _ in main[p["tag"]]["history"]]
    k, t = p["k"], p["t"]
    start = 0 if k == 0 else hist[k - 1]
    end = hist[k] if k < len(hist) else None
    p.update(el=t - start, phase=(t - start) / (end - start) if end else None)
print(f"{task}: {len(pts)} decision points in use")


def summary(g):
    Y = [p["y"] for p in g]
    same = np.mean([y[i, 0] == y[i, 1] for y in Y for i in range(len(y))])
    cross = np.mean([y[i, a] == y[j, b] for y in Y for i in range(len(y)) for j in range(len(y)) if i != j for a in (0, 1) for b in (0, 1)])
    return f"{effect(g)}  same {same:.2f} vs cross {cross:.2f}"


print("all:", summary(pts))
print("--- by position in the stage (main run finished that stage; fraction of the stage elapsed)")
for k in range(4):
    for a, b, lab in [(0, .34, "early"), (.34, .67, "mid"), (.67, 1.01, "late")]:
        g = [p for p in pts if p["k"] == k and p["phase"] is not None and a <= p["phase"] < b]
        if len(g) >= 8:
            print(f"S{k} {lab:5s}", summary(g))
print("--- by steps since the stage started (all points)")
for k in range(4):
    for a, b in [(0, 80), (80, 160), (160, 240), (240, 400), (400, 10 ** 6)]:
        g = [p for p in pts if p["k"] == k and a <= p["el"] < b]
        if len(g) >= 8:
            print(f"S{k} {a:4d}-{b if b < 10 ** 6 else '':4}", summary(g))
