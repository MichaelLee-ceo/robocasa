"""Vectorised evaluation of gripper-ramp inference rules on out/grip_table.npz (see grip_table.py, grip_rules3.py).
Rules output c in [-1, 1] (+1 = close). g0 = gap now, gN = gap N steps earlier.
  a_v3     closing iff g0 < 75.17 mm and not g0-g1 > 0.5 mm                       (rows up to settled-v3)
  b_v4     closing iff g0 < 78.5 mm and not g0-g1 > 0.5 mm                        (hidden_state.py now)
  c_v5     d8 = g0-g8: closing if d8 < -0.3 mm, opening if > +0.3 mm, else closing iff g0 < 78.5 mm
  d_val    opening if g0-g4 > 0.2 mm, closing if < -0.2 mm; steady: object contact -> +1, else 1 - g0/40 mm (0.2 grid)
  f_hyb    held (both fingers on one free object): b_v4; otherwise a_v3
  f_hyb_tr f_hyb, and when not held: closing if d8 < -1 mm, opening if d8 > +1 mm
  oracle   sign of the true c (the best any +-1 rule can do)
usage: python -I grip_eval.py -> out/grip_eval.txt"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
OPEN_V4, OPEN_V3 = 0.0785, abs(0.020833 * 2) * 1.9 - 0.004


def rules(T):
    h = T["hist"]
    g0, g1, g4, g8 = h[:, -1], h[:, -2], h[:, -5], h[:, -9]
    v1, d4, d8 = g0 - g1, g0 - g4, g0 - g8
    held, obj = T["held"], T["obj"]
    R = {}
    R["a_v3"] = np.where((g0 >= OPEN_V3) | (v1 > 0.0005), -1.0, 1.0)
    R["b_v4"] = np.where((g0 >= OPEN_V4) | (v1 > 0.0005), -1.0, 1.0)
    R["c_v5"] = np.where(d8 < -0.0003, 1.0, np.where(d8 > 0.0003, -1.0, np.where(g0 < OPEN_V4, 1.0, -1.0)))
    val = np.clip(np.round((1 - g0 / 0.04) / 0.2) * 0.2, -1, 1)
    R["d_val"] = np.where(d4 > 0.0002, -1.0, np.where(d4 < -0.0002, 1.0, np.where(obj, 1.0, val)))
    R["f_hyb"] = np.where(held, R["b_v4"], R["a_v3"])
    tr = np.where(d8 < -0.001, 1.0, np.where(d8 > 0.001, -1.0, R["a_v3"]))
    R["f_hyb_tr"] = np.where(held, R["b_v4"], tr)
    R["oracle"] = np.where(T["c"] >= 0, 1.0, -1.0)
    return R


def situations(T):
    g0 = T["hist"][:, -1]
    c, held, obj, fix = T["c"], T["held"], T["obj"], T["fix"]
    mid = np.abs(c) < 0.999
    steady = np.abs(g0 - T["hist"][:, -5]) < 0.0002
    lab = np.full(len(c), "", dtype=object)
    free = ~obj & ~fix
    rules_ = [
        (mid & (g0 >= 0.005) & (g0 < 0.075) & steady, "mid-ramp, steady mid gap"),
        (mid & (T["cmd_last"] < 0), "mid-ramp, releasing (last cmd open)"),
        (mid & (T["cmd_last"] > 0), "mid-ramp, grasping (last cmd close)"),
        (held & (g0 >= 0.060), "holding, wide grasp >= 60 mm"),
        (held & (g0 >= 0.020), "holding, 20-60 mm"),
        (held, "holding, < 20 mm"),
        (obj, "touching an object, not held"),
        (fix & (g0 >= 0.005) & (g0 < 0.075), "touching a fixture, mid gap"),
        (fix, "touching a fixture, gap < 5 / >= 75 mm"),
        (free & (g0 >= OPEN_V4), "free, >= 78.5 mm"),
        (free & (g0 >= OPEN_V3), "free, 75.2-78.5 mm (creep band)"),
        (free & (g0 < 0.005), "free, closed < 5 mm"),
        (free, "free, mid gap 5-75 mm"),
    ]
    for m, name in reversed(rules_):
        lab[m] = name
    return lab


def main():
    Z = np.load(f"{OUT}/grip_table.npz", allow_pickle=True)
    T = {k: Z[k] for k in Z.files}
    R = rules(T)
    lab = situations(T)
    names = list(R)
    c = T["c"]
    sat = np.abs(c) > 0.999
    lines = ["cells: wrong side at saturated boundaries (%) / mean |c_rule - c_true| in policy steps (one step = 0.2)"]
    for ti in [-1] + list(range(len(T["tasks"]))):
        sel_t = np.ones(len(c), bool) if ti < 0 else T["task"] == ti
        lines.append(f"\n== {'ALL TASKS' if ti < 0 else T['tasks'][ti]}  ({sel_t.sum()} boundaries, {np.mean(~sat[sel_t]):.1%} mid-ramp)")
        lines.append(f"{'situation':40s} {'n':>6s} " + " ".join(f"{n:>14s}" for n in names))
        for s in ["ALL", "saturated only", "mid-ramp only"] + sorted(set(lab)):
            m = sel_t & (np.ones(len(c), bool) if s == "ALL" else sat if s == "saturated only" else ~sat if s == "mid-ramp only" else lab == s)
            if not m.any():
                continue
            cells = []
            for n in names:
                err = np.abs(R[n][m] - c[m]) / 0.2
                ms = m & sat
                w = (np.sign(R[n][ms]) != np.sign(c[ms])).mean() if ms.any() else np.nan
                cells.append(("    -" if np.isnan(w) else f"{w:5.1%}") + f"/{err.mean():4.2f}")
            lines.append(f"{s[:40]:40s} {m.sum():6d} " + " ".join(f"{x:>14s}" for x in cells))
    # direction of the errors of the current rule and of f_hyb
    lines.append("\nerror direction (saturated boundaries, all tasks): true open read closing / true closed read opening")
    for n in names:
        lines.append(f"  {n:9s} {int(((c < -0.999) & (R[n] > 0)).sum()):6d} / {int(((c > 0.999) & (R[n] < 0)).sum()):6d}")
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/grip_eval.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
