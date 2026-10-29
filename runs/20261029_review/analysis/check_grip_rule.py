"""Which rows in use were collected with a gripper ramp that the current settle() (settled-v4) would set differently?
grip_check.py looked only at points where the main run's fingers touch an object. The two rules differ wherever the
finger gap at t is between the old "fully open" threshold (1.9 x init gap - 4 mm = 75.2 mm) and the new one (joint
limits - 1.5 mm = 78.5 mm) and the fingers are not widening: old -> "opening", new -> "closing". That also covers
fingers closing in the air. Gap from the main run's trace (grip_width = sum of |finger qpos|, 0.1 mm). Read-only.
usage: python -I check_grip_rule.py"""
import collections, glob, gzip, json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
SEEN = f"{R}/20261025_seen480"
sys.path.insert(0, f"{R}/20261028_critical")
from effect_by_event import RUNS, points  # noqa: E402

OLD, NEW = 0.0792 - 0.004, 0.080 - 0.0015


def infer(gap, prev, thr):
    closing = gap < thr
    if prev is not None and closing:
        closing = not (gap - prev > 0.0005)
    return "closing" if closing else "opening"


def main():
    tr = {}
    def gap(tag, task, t):
        if tag not in tr:
            tr[tag] = json.load(gzip.open(f"{SEEN}/analysis/{task}/{tag}_trace.json.gz"))["grip_width"]
        g = tr[tag]
        return g[t], (g[t - 1] if t else None)
    rows = [("coarse in use", p["task"], p["tag"], p["t"], p["hidden"]) for p in points()]
    for task in RUNS:
        for d in (f"run_dense_{task}", f"run_dense2_{task}", f"run_confirm_{task}"):
            for f in glob.glob(f"{DATA}/{d}/results_w*.jsonl"):
                for line in open(f):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" not in r:
                        rows.append((d.replace(task, "<task>"), task, r["tag"], r["t"], r.get("hidden")))
    c = collections.Counter(); ex = collections.defaultdict(list)
    for kind, task, tag, t, hidden in rows:
        g, gp = gap(tag, task, t)
        o, n = infer(g, gp, OLD), infer(g, gp, NEW)
        used = n if hidden == "settled-v4" else o
        diff = used != n
        c[(kind, task, diff)] += 1
        if diff:
            ex[(kind, task)].append((tag.split("_")[1], t, round(1000 * g, 1)))
    for (kind, task, diff), n in sorted(c.items()):
        if diff:
            print(f"{kind:22s} {task:21s} rows collected with a ramp the current rule would not set: {n:3d} of "
                  f"{n + c[(kind, task, False)]}   e.g. {ex[(kind, task)][:6]}")


if __name__ == "__main__":
    main()
