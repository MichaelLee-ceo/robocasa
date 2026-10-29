"""Arm-lag check at base-mode starts: settle puts the OSC goal at the achieved pose, while in the main run the arm still
lags behind its accumulated goal ("desired" mode). Per current point (gripper-consistent classes only), the main run's
end-effector path in the base frame against the 8 candidates': |main - candidate median| and the main's distance to
the nearest candidate at steps 1, 2, 4, 8, 16, by start mode. usage: python -I eef_lag.py"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402

R = pd.read_csv(os.path.join(HERE, "ramp_classes.csv.gz"))
ok = {(r.tag, r.t, r.dir, r.role) for r in R.itertuples() if r.cls[0] in "FG"}
rows = []
for d in metrics.load():
    mn = d["main"]
    for p in d["pts"]:
        if p["role"] not in ("cur", "dense", "dense2", "confirm") or (d["tag"], p["t"], p["dir"].split("/")[0], p["role"]) not in ok:
            continue
        t = p["t"]
        m0 = [r for r in p["rolls"] if r["m"] == 0]
        n = min([r["T"] for r in m0] + [17, mn["T"] - t])
        if len(m0) < 8 or n < 17:
            continue
        E = np.stack([r["eef_b"][:n] - r["eef_b"][0] for r in m0])
        em = mn["eef_b"][t:t + n] - mn["eef_b"][t]
        rec = dict(task=d["task"], tag=d["tag"], t=t, dir=p["dir"].split("/")[0], mode0=float(np.mean([r["a_control_mode"][0, 0, 0] > 0.5 for r in m0])))
        for s in (1, 2, 4, 8, 16):
            med = np.median(E[:, s], 0)
            rec[f"dev{s}"] = 1000 * float(np.linalg.norm(em[s] - med))
            rec[f"near{s}"] = 1000 * float(np.min(np.linalg.norm(E[:, s] - em[s], axis=1)))
            rec[f"spread{s}"] = 1000 * float(np.median(np.linalg.norm(E[:, s] - med, axis=1)))
        rows.append(rec)
df = pd.DataFrame(rows)
df["start"] = np.where(df.mode0 >= 0.5, "base-mode start", "arm-mode start")
pd.set_option("display.width", 250)
cols = [f"{k}{s}" for s in (1, 4, 16) for k in ("dev", "near", "spread")]
print("medians (mm): main vs candidate median (dev), main vs nearest candidate (near), candidates' own median spread")
print(df.groupby(["task", "start"])[cols].median().round(1).to_string())
print("90th percentiles (mm)")
print(df.groupby(["task", "start"])[cols].quantile(0.9).round(1).to_string())
print(df.groupby(["task", "start"]).size())
df.to_csv(os.path.join(HERE, "eef_lag.csv.gz"), index=False)
