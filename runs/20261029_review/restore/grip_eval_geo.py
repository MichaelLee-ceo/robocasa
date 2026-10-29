"""Adds the contact-side rule to grip_eval.py's table (out/grip_geo.json from grip_geo.py).
  g_geo   held -> b_v4; fingers touching a fixture (no free object) with a 5-75.2 mm gap -> closing if more finger
          contacts push the fingers open (something between them) than closed, opening if the reverse (squeezed),
          a_v3 on a tie; everything else -> a_v3   (= f_hyb + the contact-side test on fixtures)
  g_geo2  same, the test also for non-held object contacts
  g_final g_geo with held -> closing iff gap < 78.5 mm (settle_candidate.py)
usage: python -I grip_eval_geo.py -> out/grip_eval_geo.txt"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_eval import rules, situations

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    Z = np.load(f"{OUT}/grip_table.npz", allow_pickle=True)
    T = {k: Z[k] for k in Z.files}
    geo = json.load(open(f"{OUT}/grip_geo.json"))
    R = rules(T); lab = situations(T)
    nin = np.array([geo.get(k, (0, 0))[0] for k in T["key"]]); nout = np.array([geo.get(k, (0, 0))[1] for k in T["key"]])
    g0 = T["hist"][:, -1]
    mid = (g0 >= 0.005) & (g0 < 0.07517)  # below the v3 open threshold (a_v3 is right above it)
    side = np.where(nin > nout, 1.0, np.where(nout > nin, -1.0, 0.0))
    for name, where in (("g_geo", T["fix"] & ~T["obj"] & ~T["held"] & mid), ("g_geo2", (T["fix"] | T["obj"]) & ~T["held"] & mid)):
        base = R["f_hyb"].copy()
        use = where & (side != 0)
        base[use] = side[use]
        R[name] = base
    # final candidate (settle_candidate.py): g_geo, but a held object -> closing iff the gap is below 78.5 mm (no 1-step
    # widening test: a held object often pushes the fingers apart by > 0.5 mm in one step)
    gf = R["g_geo"].copy()
    gf[T["held"]] = np.where(g0[T["held"]] < 0.0785, 1.0, -1.0)
    R["g_final"] = gf
    names = ["a_v3", "b_v4", "c_v5", "d_val", "f_hyb", "g_geo", "g_final", "oracle"]
    c = T["c"]; sat = np.abs(c) > 0.999
    lines = ["cells: wrong side at saturated boundaries (%) / mean |c_rule - c_true| in policy steps"]
    for ti in [-1] + list(range(len(T["tasks"]))):
        sel_t = np.ones(len(c), bool) if ti < 0 else T["task"] == ti
        lines.append(f"\n== {'ALL TASKS' if ti < 0 else T['tasks'][ti]} ({sel_t.sum()} boundaries)")
        lines.append(f"{'situation':40s} {'n':>6s} " + " ".join(f"{n:>13s}" for n in names))
        for s in ["ALL", "mid-ramp only"] + sorted(set(lab)):
            m = sel_t & (np.ones(len(c), bool) if s == "ALL" else ~sat if s == "mid-ramp only" else lab == s)
            if not m.any():
                continue
            cells = []
            for n in names:
                err = np.abs(R[n][m] - c[m]) / 0.2
                ms = m & sat
                w = (np.sign(R[n][ms]) != np.sign(c[ms])).mean() if ms.any() else np.nan
                cells.append(("    -" if np.isnan(w) else f"{w:5.1%}") + f"/{err.mean():4.2f}")
            lines.append(f"{s[:40]:40s} {m.sum():6d} " + " ".join(f"{x:>13s}" for x in cells))
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/grip_eval_geo.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
