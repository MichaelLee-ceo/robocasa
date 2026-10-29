"""Gripper-ramp classes per point (from ramp.csv.gz, ramp.py) and the list of current points whose restored start is
still perturbed (corrupted_points.csv).

Classes (the main run at t, see ramp.py for the columns):
  A t = 0                         the true ramp is 0 (GripperModel starts with current_action = 0); settle sets +1
  B v4: fully open read closing   settled-v4, free fingers, gap above the old open threshold, inferred "close"
  C gripper moving, touching      |gap change t-1 -> t| > 0.5 mm, an object touched (trace)
  D gripper moving, free          the same without contact
  E static mid gap, free          free fingers at an intermediate gap (ramp in between, alternating commands)
  F touching, read as opening
  G static, settled side plausible
A point is perturbed when the main run's finger gap lies > 3 mm outside the range of all 8 candidates at step 2 or 4.
Severity adds whether the main run touches an object within the first chunk (t .. t + 16, its trace).
usage: python -I classify.py"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402

OLD_OPEN = 0.020833 * 2 * 1.9


def cls(r):
    if r.t == 0:
        return "A t=0 (true ramp 0)"
    g = r.gap0 / 1000
    moving = abs(r.dgap_prev) > 0.5 if pd.notna(r.dgap_prev) else False
    if r.hidden == "settled-v4" and not r.touch and g >= OLD_OPEN - 0.004 and r.side == "close":
        return "B v4: fully open read as closing"
    if moving and r.touch:
        return "C gripper moving, touching"
    if moving:
        return "D gripper moving, free"
    if not r.touch and 0.005 < g < OLD_OPEN - 0.004:
        return "E static mid gap, free"
    if r.touch and r.side == "open":
        return "F touching, read as opening"
    return "G static, settled side plausible"


def main():
    R = pd.read_csv(os.path.join(HERE, "ramp.csv.gz"))
    R["cls"] = R.apply(cls, axis=1)
    R.to_csv(os.path.join(HERE, "ramp_classes.csv.gz"), index=False)
    c = R[R.role.isin(["cur", "dense", "dense2", "confirm"])].copy()
    c["out"] = c[["out2", "out4"]].max(axis=1)
    c = c[c.out > 3].copy()
    touch16 = {}
    for tag in sorted(set(c.tag)):
        d = next(metrics.load(tag))
        mn = d["main"]
        for t in set(c[c.tag == tag].t):
            tt = mn["trace_touch"][t:t + 17]
            touch16[(tag, t)] = (bool(tt.any()), ",".join(d["objs"][i] for i in np.where(tt.any(0))[0]))
    c["main_touch_within_16"] = [touch16[(r.tag, r.t)][0] for r in c.itertuples()]
    c["objs_touched"] = [touch16[(r.tag, r.t)][1] for r in c.itertuples()]

    def sev(r):
        if r.cls.startswith("A"):
            return "A t0: transient spurious closing, nothing in reach"
        if r.cls.startswith("B"):
            return "B v4 regression: open gripper snaps closed"
        if r.main_touch_within_16:
            return "C/D/E transition with an object in contact within the chunk: label may change"
        return "D/E free-gripper transition, no contact within the chunk: transient"
    c["severity"] = c.apply(sev, axis=1)
    c["task"] = c.task.map({"KettleBoiling": "KB", "SteamInMicrowave": "SIM", "PackIdenticalLunches": "PIL"})
    c["tag"] = c.tag.str.replace("_single", "")
    cols = ["task", "tag", "t", "k", "dir", "role", "hidden", "cls", "severity", "gap0", "dgap_prev", "touch", "objs_touched",
            "r_est", "side", "cmd_close0", "dev1", "dev2", "dev4", "dev8", "out", "n_close_cmd_but_open", "done"]
    c[cols].sort_values(["severity", "task", "tag", "t"]).to_csv(os.path.join(HERE, "corrupted_points.csv"), index=False)
    cur = R[R.role.isin(["cur", "dense", "dense2", "confirm"])]
    print(f"{len(cur)} current point instances, {len(c)} perturbed")
    print(c.groupby(["severity", "task"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
