import json, sys
import numpy as np
R_DIR = "/home/michaellee/mclee/robocasa/runs/20261029_review/restore"
Z = np.load(f"{R_DIR}/out/grip_table.npz", allow_pickle=True); T = {k: Z[k] for k in Z.files}
h = T["hist"]; g0, g1, g2 = h[:, -1], h[:, -2], h[:, -3]; c = T["c"]
OPEN_V3 = abs(0.020833 * 2) * 1.9 - 0.004
free = ~T["obj"] & ~T["fix"] & ~T["held"]
mid = (g0 >= 0.005) & (g0 < OPEN_V3)
v1, v2 = g0 - g1, g1 - g2
grid = lambda x: np.clip(np.round(x / 0.2) * 0.2, -1, 1)
sel = free & mid
print("free & mid:", sel.sum(), " saturated c:", (sel & (abs(c) > .999)).sum())
# fit c ~ 1 - (g0 + k1*v1 + k2*v2)/0.04 on free & mid
X = np.stack([v1[sel], v2[sel]], 1) / 0.04
y = (1 - g0[sel] / 0.04) - c[sel]          # = -(k1 v1 + k2 v2)/0.04
for cols, nm in (([0], "v1"), ([0, 1], "v1,v2")):
    k, *_ = np.linalg.lstsq(-X[:, cols], y, rcond=None)
    print(nm, "coef", np.round(k, 3))
def score(est, nm):
    e = np.abs(est[sel] - c[sel]) / 0.2
    sat = np.abs(c[sel]) > .999
    print(f"{nm:22s} mean {e.mean():.2f}  >=1 step {np.mean(e >= .999):5.1%}  >=2 {np.mean(e >= 1.999):5.1%} | mid-ramp only mean {e[~sat].mean():.2f} >=2 {np.mean(e[~sat] >= 1.999):5.1%} | saturated mean {e[sat].mean():.2f} wrong side {np.mean(np.sign(est[sel][sat]) != np.sign(c[sel][sat])):.1%}")
for kk in (0, 1, 1.5, 2, 2.5, 3, 4):
    score(grid(1 - (g0 + kk * v1) / 0.04), f"g0 + {kk}*v1")
for lag in (1, 2):
    score(grid(1 - g0 / 0.04 + lag * 0.2 * np.sign(-v1) * (np.abs(v1) > 0.0003)), f"lag {lag} steps")
k, *_ = np.linalg.lstsq(-X, y, rcond=None)
score(grid(1 - (g0 + k[0] * v1 + k[1] * v2) / 0.04), f"fit v1,v2 {np.round(k,2)}")
# baseline g_final there = a_v3 (free, mid) -> +1 unless widening
score(np.where(v1 > 0.0005, -1.0, 1.0), "g_final (a_v3)")
