import json, sys
import numpy as np
R_DIR = "/home/michaellee/mclee/robocasa/runs/20261029_review/restore"
sys.path.insert(0, R_DIR)
from grip_eval import rules, situations
Z = np.load(f"{R_DIR}/out/grip_table.npz", allow_pickle=True); T = {k: Z[k] for k in Z.files}
geo = json.load(open(f"{R_DIR}/out/grip_geo.json"))
R0 = rules(T); lab = situations(T)
nin = np.array([geo.get(k, (0, 0))[0] for k in T["key"]]); nout = np.array([geo.get(k, (0, 0))[1] for k in T["key"]])
h = T["hist"]; g0, g1 = h[:, -1], h[:, -2]; v1 = g0 - g1; c = T["c"]
OPEN_V3 = abs(0.020833 * 2) * 1.9 - 0.004
mid = (g0 >= 0.005) & (g0 < OPEN_V3)
side = np.where(nin > nout, 1.0, np.where(nout > nin, -1.0, 0.0))
gf = R0["f_hyb"].copy(); use = T["fix"] & ~T["obj"] & ~T["held"] & mid & (side != 0); gf[use] = side[use]
gf[T["held"]] = np.where(g0[T["held"]] < 0.0785, 1.0, -1.0)
grid = lambda x: np.clip(np.round(x / 0.2) * 0.2, -1, 1)
ex = grid(1 - (g0 + 2 * v1) / 0.04)
free = ~T["obj"] & ~T["fix"] & ~T["held"]
objonly = T["obj"] & ~T["held"]
R = {"g_final": gf, "oracle": R0["oracle"]}
r = gf.copy(); r[free & mid] = ex[free & mid]; R["X_freemid"] = r
r = gf.copy(); r[free] = ex[free]; R["X_free"] = r
r = gf.copy(); r[free | (objonly & ~T["fix"])] = ex[free | (objonly & ~T["fix"])]; R["X_free+obj"] = r
sat = np.abs(c) > 0.999
bands = [("free <5mm", free & (g0 < 0.005)), ("free 5-75.2", free & mid), ("free 75.2-78.5", free & (g0 >= OPEN_V3) & (g0 < 0.0785)),
         ("free >=78.5", free & (g0 >= 0.0785)), ("obj not held", objonly), ("fixture only", T["fix"] & ~T["obj"] & ~T["held"]), ("held", T["held"])]
print("cells: n | wrong side @saturated | mean err (steps) | >=2 steps")
for ti in [-1, 0, 1, 2]:
    st = np.ones(len(c), bool) if ti < 0 else T["task"] == ti
    print(f"\n== {'ALL' if ti < 0 else T['tasks'][ti]}")
    for nm, m0 in [("ALL", np.ones(len(c), bool)), ("mid-ramp", ~sat)] + bands:
        m = st & m0
        if not m.any():
            continue
        cells = []
        for n in R:
            e = np.abs(R[n][m] - c[m]) / 0.2; ms = m & sat
            w = (np.sign(R[n][ms]) != np.sign(c[ms])).mean() if ms.any() else np.nan
            cells.append(f"{w:5.1%}/{e.mean():4.2f}/{np.mean(e >= 1.999):5.1%}")
        print(f"{nm:16s} {m.sum():6d} " + " ".join(f"{n}={x}" for n, x in zip(R, cells)))
