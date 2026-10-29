"""Ramp estimators for the "fixture contact, fingers moving" branch, scored on the restore review's ground truth
(../restore/out/grip_table.npz: true ramp c at chunk boundaries of saved rollouts). Future finger gaps are read from the
saved rollouts. At a decision point the main run's future states are known too, but not its commands; the estimators
here use only the gaps, plus the direction of the motion.
  past    v5: c = 1 - (g0 + 2 (g0 - g-1)) / 40 mm
  future  target during the next step = g0 + 2 (g+1 - g0); c = 1 - target / 40 mm - 0.2 s, s = +1 if closing (g+1 < g0)
  faster  the one of the two whose motion is larger (contact can only slow the fingers)
  old     +-1: closing iff g0 < 75.2 mm and not widened > 0.5 mm
usage: python -I fixture_future.py"""
import os, sys
from collections import defaultdict
import numpy as np
RV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "restore")
sys.path.insert(0, RV)
import harness as H  # noqa: E402

Z = np.load(f"{RV}/out/grip_table.npz", allow_pickle=True); T = {k: Z[k] for k in Z.files}
g0, g1 = T["hist"][:, -1], T["hist"][:, -2]; v1 = g0 - g1
sel = np.nonzero(T["fix"] & ~T["obj"] & ~T["held"] & (np.abs(v1) > 0.0005))[0]
print("fixture-only & moving boundaries:", len(sel))
by_file = defaultdict(list)
for i in sel:
    d, tag, t, c, b = T["key"][i].split("|")
    by_file[(d, tag, int(t), int(c))].append((i, int(b)))
fut = {}
for (d, tag, t, c), lst in by_file.items():
    try:
        s = np.load(H.roll_path(d, tag, t, c, 0))["states"]
    except Exception:
        continue
    g = np.abs(s[:, 12] - s[:, 13])
    for i, b in lst:
        if b + 1 < len(g):
            fut[i] = g[b + 1]
idx = np.array(sorted(fut)); gp1 = np.array([fut[i] for i in idx])
c = T["c"][idx]; G0, V1 = g0[idx], v1[idx]
grid = lambda x: np.clip(np.round(x / 0.2) * 0.2, -1, 1)
past = grid(1 - (G0 + 2 * V1) / 0.04)
s = np.where(gp1 < G0, 1.0, -1.0)
future = grid(1 - (G0 + 2 * (gp1 - G0)) / 0.04 - 0.2 * s)
faster = np.where(np.abs(gp1 - G0) > np.abs(V1), future, past)
old = np.where((G0 >= 0.07517) | (V1 > 0.0005), -1.0, 1.0)
sat = np.abs(c) > 0.999
for name, est in (("past (v5)", past), ("future", future), ("faster", faster), ("old +-1", old)):
    e = np.abs(est - c) / 0.2
    print(f"{name:10s} n {len(e)}  mean err {e.mean():.2f} steps  >=2 steps {np.mean(e >= 1.999):5.1%}  wrong side @sat {np.mean(np.sign(est[sat]) != np.sign(c[sat])):5.1%}  mid-ramp mean {e[~sat].mean() if (~sat).any() else float('nan'):.2f}")
