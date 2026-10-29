"""Where along a trajectory does the outcome get decided? The value curve of the main runs from the collected points.

At every collected decision point, the 16 rollouts all start from the main run's state at t, so their completion rate
of stage k estimates V(t) = P(finish the current stage | state at t, plain policy). The chunk the outcome hinges on lies
where V jumps between consecutive points of the same stage: a drop (e.g. a missed grasp) or a rise (e.g. a grasp that
holds). This does not depend on any task knowledge. The jumps found here are intervals of 5-20 chunks: KettleBoiling
points are 80 steps apart, SteamInMicrowave 160 and PackIdenticalLunches 320. Dense points inside them locate the
chunk itself.
Also: whether the main run's own physical events (events.py) fall inside the jump intervals, as a check that the jumps
mark real decisions. The chunks between t0 and t1 produce the states t0+1..t1, so an interval's events are those at
steps (t0, t1]. Points: the coarse rows in use (data.py). Writes value_jumps.jsonl to $CRIT_OUT.
usage: python -I value_curve.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import HERE, OUT, TASKS  # noqa: E402
from effect_by_event import points  # noqa: E402

JUMP = 0.4


def main():
    ev = {r["tag"]: r for r in map(json.loads, open(os.path.join(HERE, "events.jsonl")))}
    by = {}
    for p in points():
        by.setdefault(p["tag"], []).append(p)
    rows = []
    for tag, g in by.items():
        g.sort(key=lambda p: p["t"])
        for a, b in zip(g[:-1], g[1:]):
            if a["k"] == b["k"]:
                rows.append(dict(task=a["task"], tag=tag, k=a["k"], t0=a["t"], t1=b["t"], v0=a["pd"], v1=b["pd"], dv=b["pd"] - a["pd"]))
    dv = np.array([r["dv"] for r in rows])
    print(f"{len(rows)} consecutive same-stage pairs of points")
    for task in TASKS:
        g = np.array([r["dv"] for r in rows if r["task"] == task])
        print(f"  {task:21s} pairs {len(g):3d}: |dV| >= {JUMP}: drops {np.sum(g <= -JUMP):3d}, rises {np.sum(g >= JUMP):3d}; "
              f"|dV| < 0.1: {np.mean(np.abs(g) < 0.1):.0%}")
    print(f"  all: drops {np.sum(dv <= -JUMP)}, rises {np.sum(dv >= JUMP)}, flat (|dV| < 0.1) {np.mean(np.abs(dv) < 0.1):.0%}")
    print("--- events of the main run inside the jump intervals vs inside flat intervals (per interval, any event of the kind)")
    kinds = {"grasp (hold, close)": {"hold", "close"}, "release (release, open)": {"release", "open"}, "fixture (fx_touch, fx_move)": {"fx_touch", "fx_move"}}
    for lab, cond in [("drop", lambda r: r["dv"] <= -JUMP), ("rise", lambda r: r["dv"] >= JUMP), ("flat", lambda r: abs(r["dv"]) < 0.1)]:
        g = [r for r in rows if cond(r)]
        frac = {k: np.mean([any(e["type"] in ts and r["t0"] < e["t"] <= r["t1"] for e in ev[r["tag"]]["events"]) for r in g]) for k, ts in kinds.items()}
        print(f"  {lab:4s} n {len(g):3d}: " + ", ".join(f"{k} {v:.0%}" for k, v in frac.items()))
    print("--- examples (largest drops and rises)")
    for r in sorted(rows, key=lambda r: r["dv"])[:6] + sorted(rows, key=lambda r: -r["dv"])[:4]:
        evs = [f"{e['t']}:{e['type']}" for e in ev[r["tag"]]["events"] if r["t0"] < e["t"] <= r["t1"] and e["type"] not in ("reach", "touch")]
        print(f"  {r['tag']:31s} S{r['k']} t {r['t0']:4d}->{r['t1']:4d}  V {r['v0']:.2f} -> {r['v1']:.2f}   events in between: {' '.join(evs[:8])}")
    with open(os.path.join(OUT, "value_jumps.jsonl"), "w") as f:
        for r in rows:
            if abs(r["dv"]) >= JUMP:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
