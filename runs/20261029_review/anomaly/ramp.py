"""Gripper-ramp check: how far the restored candidates' finger gap departs from the main run's in the first steps, and
whether the settle rule's guess of the commanded side explains it.

The Panda fingers are position-controlled: target gap = 0.04 * (1 - r) m, r = gripper.current_action[1] in [-1, 1]
(+1 close), moved by 0.2 per step towards the commanded side (PandaGripper.format_action). settle() sets r to +1 or -1
only. With free fingers the gap tracks the target, so r_est = 1 - gap / 0.04 estimates the true r (at t = 0 the true r
is 0: GripperModel starts with current_action = 0).
Per collected point (m0 of the 8 candidates; current and superseded rows), reports:
  side   the side the row's settle rule inferred (old rule for settled / basefix / v3 rows, new rule for v4)
  dev_s  main gap - median candidate gap at step s (mm); out_s: how far the main lies outside the candidates' range
  plus touch / grasp at t (trace), the main's gap change t-1 -> t, and the candidates' gripper commands.
Writes ramp.csv.gz. usage: python -I ramp.py"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402

OPEN_OLD = 0.020833 * 2 * 1.9  # PandaGripper init gap x 1.9 (rule before bug 4)
OPEN_NEW = 0.08  # finger joint limits


def inferred(gap, gap_prev, hidden):
    if hidden == "settled-v4":
        closing = gap < OPEN_NEW - 0.0015
    else:
        closing = gap < OPEN_OLD - 0.004
    if gap_prev is not None and closing:
        closing = not (gap - gap_prev > 0.0005)
    return "close" if closing else "open"


def main():
    rows = []
    for d in metrics.load():
        mn = d["main"]
        for p in d["pts"]:
            t = p["t"]
            m0 = [r for r in p["rolls"] if r["m"] == 0]
            if len(m0) < 8 or t + 9 > mn["T"]:
                continue
            n = min([r["T"] for r in m0] + [17, mn["T"] - t])
            G = np.array([r["gap"][:n] for r in m0])
            gm = mn["gap"][t:t + n]
            rec = dict(task=d["task"], tag=d["tag"], t=t, k=p["k"], dir=p["dir"].split("/")[0], role=p["role"], hidden=p["hidden"],
                       gap0=round(1000 * float(mn["gap"][t]), 1),
                       dgap_prev=round(1000 * float(mn["gap"][t] - mn["gap"][t - 1]), 2) if t else None,
                       side=inferred(float(mn["gap"][t]), float(mn["gap"][t - 1]) if t else None, p["hidden"]),
                       touch=bool(mn["trace_touch"][t].any()), grasp=bool(mn["trace_grasp"][t].any()),
                       r_est=round(float(np.clip(1 - mn["gap"][t] / 0.04, -1, 1)), 2),
                       cmd_close0=float(np.mean([r["a_gripper_close"][0, 0, 0] > 0.5 for r in m0])),
                       cmd_close4=float(np.mean([(r["a_gripper_close"][0, :4, 0] > 0.5).mean() for r in m0])),
                       done=float(np.mean([r["done"] for r in m0])))
            for s in (1, 2, 4, 8, 16):
                if s < n:
                    rec[f"dev{s}"] = round(1000 * float(gm[s] - np.median(G[:, s])), 2)
                    lo, hi = G[:, s].min(), G[:, s].max()
                    rec[f"out{s}"] = round(1000 * float(max(gm[s] - hi, lo - gm[s], 0)), 2)
                    rec[f"spread{s}"] = round(1000 * float(hi - lo), 2)
            # candidates that command close for their first 4 steps yet open > 5 mm within 8 steps while touching
            nopen = 0
            for r in m0:
                if r["touch"][0].any() and (r["a_gripper_close"][0, :4, 0] > 0.5).all():
                    if np.max(r["gap"][1:min(9, r["T"])]) - r["gap"][0] > 0.005:
                        nopen += 1
            rec["n_close_cmd_but_open"] = nopen
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "ramp.csv.gz"), index=False)
    print(len(df), "point instances")


if __name__ == "__main__":
    main()
