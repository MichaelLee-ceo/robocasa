import json, os, sys
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrap_all.py")).read().split("sat = np.abs")[0])
sat = np.abs(c) > 0.999
fixonly = T["fix"] & ~T["obj"] & ~T["held"]
base = R["X_free+obj"]
V = {"X": base}
for thr in (0.0005, 0.001, 0.002):
    r = base.copy(); s = fixonly & (np.abs(v1) > thr); r[s] = ex[s]; V[f"X+fixmove{thr*1000:g}"] = r
r = base.copy(); r[fixonly] = ex[fixonly]; V["X+fix_all"] = r
V["oracle"] = R0["oracle"]
for nm, m in [("ALL", np.ones(len(c), bool)), ("mid-ramp", ~sat), ("fixture only", fixonly), ("fix & mid-ramp", fixonly & ~sat), ("fix & sat", fixonly & sat),
              ("held & mid-ramp", T["held"] & ~sat), ("obj-not-held & mid-ramp", objonly & ~sat), ("free & mid-ramp", free & ~sat)]:
    cells = []
    for n in V:
        e = np.abs(V[n][m] - c[m]) / 0.2; ms = m & sat
        w = (np.sign(V[n][ms]) != np.sign(c[ms])).mean() if ms.any() else np.nan
        cells.append(f"{n}={w:5.1%}/{e.mean():4.2f}/{np.mean(e >= 1.999):5.1%}")
    print(f"{nm:24s} {m.sum():6d} " + " ".join(cells))
# per task for fixture only
for ti in range(3):
    m = fixonly & (T["task"] == ti)
    print(T["tasks"][ti], m.sum(), {n: f"{(np.sign(V[n][m & sat]) != np.sign(c[m & sat])).mean():.1%}/{(np.abs(V[n][m]-c[m])/0.2).mean():.2f}" for n in V})
