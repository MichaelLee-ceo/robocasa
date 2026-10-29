"""Offline: g_final (settle_candidate) vs hybrids that use the ramp value for free fingers; on the restore review's
ground-truth table (true ramp c at 77.5k chunk boundaries)."""
import json, sys
import numpy as np
R_DIR = "/home/michaellee/mclee/robocasa/runs/20261029_review/restore"
sys.path.insert(0, R_DIR)
from grip_eval import rules, situations
OUT = f"{R_DIR}/out"
Z = np.load(f"{OUT}/grip_table.npz", allow_pickle=True); T = {k: Z[k] for k in Z.files}
geo = json.load(open(f"{OUT}/grip_geo.json"))
R = rules(T); lab = situations(T)
nin = np.array([geo.get(k, (0, 0))[0] for k in T["key"]]); nout = np.array([geo.get(k, (0, 0))[1] for k in T["key"]])
h = T["hist"]; g0, g1, g2 = h[:, -1], h[:, -2], h[:, -3]
mid = (g0 >= 0.005) & (g0 < 0.07517)
side = np.where(nin > nout, 1.0, np.where(nout > nin, -1.0, 0.0))
base = R["f_hyb"].copy(); use = T["fix"] & ~T["obj"] & ~T["held"] & mid & (side != 0); base[use] = side[use]
gf = base.copy(); gf[T["held"]] = np.where(g0[T["held"]] < 0.0785, 1.0, -1.0)
R = {"b_v4": R["b_v4"], "g_final": gf, "oracle": R["oracle"]}
free = ~T["obj"] & ~T["fix"] & ~T["held"]
grid = lambda x: np.clip(np.round(x / 0.2) * 0.2, -1, 1)
val = grid(1 - g0 / 0.04)
v1 = g0 - g1
for name, lag in (("h_val", 0.0), ("h_val_lag1", 1.0), ("h_val_lag2", 2.0)):
    for thr in (0.0003, 0.001):
        moving = np.abs(v1) > thr
        est = grid(1 - g0 / 0.04 + lag * 0.2 * np.sign(-v1) * moving)
        r = gf.copy(); sel = free & mid & moving; r[sel] = est[sel]
        R[f"{name}_m{thr*1000:g}"] = r
        r = gf.copy(); sel = free & mid; r[sel] = est[sel]
        R[f"{name}_all{thr*1000:g}"] = r
c = T["c"]; sat = np.abs(c) > 0.999
names = list(R)
print("cells: wrong side at saturated (%) / mean |c_rule - c_true| (steps of 0.2) / share with error >= 2 steps")
for ti in [-1, 0, 1, 2]:
    sel_t = np.ones(len(c), bool) if ti < 0 else T["task"] == ti
    print(f"\n== {'ALL' if ti < 0 else T['tasks'][ti]} ({sel_t.sum()})")
    print(f"{'situation':36s} {'n':>6s} " + " ".join(f"{n[:16]:>17s}" for n in names))
    for s in ["ALL", "mid-ramp only", "free, mid gap 5-75 mm", "mid-ramp, grasping (last cmd close)", "mid-ramp, releasing (last cmd open)"]:
        m = sel_t & (np.ones(len(c), bool) if s == "ALL" else ~sat if s == "mid-ramp only" else lab == s)
        if not m.any():
            continue
        cells = []
        for n in names:
            err = np.abs(R[n][m] - c[m]) / 0.2
            ms = m & sat
            w = (np.sign(R[n][ms]) != np.sign(c[ms])).mean() if ms.any() else np.nan
            cells.append(("   -" if np.isnan(w) else f"{w:4.1%}") + f"/{err.mean():4.2f}/{(err >= 1.999).mean():4.1%}")
        print(f"{s[:36]:36s} {m.sum():6d} " + " ".join(f"{x:>17s}" for x in cells))
# where does c sit for free mid-gap boundaries vs the value estimate
m = free & mid
print("\nfree & mid gap:", m.sum(), "| c saturated:", (m & sat).sum(), "| true c - value est (steps): ",
      np.unique(np.round((c[m] - val[m]) / 0.2).astype(int), return_counts=True))
mv = m & (np.abs(v1) > 0.0003)
d = np.round((c[mv] - val[mv]) / 0.2).astype(int)
print("free & mid & moving:", mv.sum(), "c - val by motion: closing", np.unique(d[v1[mv] < 0], return_counts=True), " opening", np.unique(d[v1[mv] > 0], return_counts=True))
