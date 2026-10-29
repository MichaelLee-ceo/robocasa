"""Event index semantics. Event step f (events.py) indexes the main run's saved states: f = the state after f steps, where
the condition first holds. The candidate chunk applies actions t .. t+15 and produces states t+1 .. t+16, so an event it
can cause has f in [t+1, t+16]. effect_by_event.py's "in the chunk" bin uses f - t in [0, 16): it includes events already
true at the decision state (f = t, caused by the main run's previous chunk) and leaves out f = t+16. Same convention in
value_curve.py and dense_analysis.py ([t0, t1) instead of (t0, t1]).
This counts the points whose next event sits on a boundary and recomputes the bins both ways. Read-only.
usage: python -I check_event_window.py"""
import os, sys

R = "/home/michaellee/mclee/robocasa/runs"
sys.path.insert(0, f"{R}/20261028_critical")
import numpy as np  # noqa: E402
from effect_by_event import KINDS, points, summary  # noqa: E402


def next_dt(p, types, lo):
    dts = [e["t"] - p["t"] for e in p["events"] if e["type"] in types and e["t"] - p["t"] >= lo]
    return min(dts) if dts else None


def main():
    pts = points()
    for kind in ["grasp (touch, hold, close)", "place (at, release, open)", "any of these"]:
        types = KINDS[kind]
        d_old = [next_dt(p, types, 0) for p in pts]
        n0 = sum(d == 0 for d in d_old); n16 = sum(d == 16 for d in d_old)
        print(f"--- {kind}: next event at the decision state itself (dt = 0): {n0}; exactly at the chunk's last state (dt = 16): {n16}")
        g_old = [p for p, d in zip(pts, d_old) if d is not None and 0 <= d < 16]
        d_new = [next_dt(p, types, 1) for p in pts]
        g_new = [p for p, d in zip(pts, d_new) if d is not None and 1 <= d <= 16]
        print(f"   in the chunk, as coded  [t, t+16):  {summary(g_old)}")
        print(f"   in the chunk, corrected (t, t+16]:  {summary(g_new)}")


if __name__ == "__main__":
    main()
