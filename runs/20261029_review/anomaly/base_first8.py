"""Base direction within the first chunk only: the main run's base displacement over its first S steps after t (robot
frame, accumulated per step) against the candidates'. Within the first chunk the main run's commands are draws from the
same policy at the same observation, so with a faithful restore the angle between the main run's and the candidates'
displacement is centred on 0 at every turned angle dyaw_ep; a frame error shows as an angle that follows dyaw_ep.
Writes base_first8.csv.gz. usage: python -I base_first8.py [S=8]"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402


def disp(b, S):
    D = np.zeros(2)
    for s in range(min(S, len(b) - 1)):
        dw = b[s + 1, :2] - b[s, :2]
        c, sn = np.cos(-b[s, 2]), np.sin(-b[s, 2])
        D += np.array([c * dw[0] - sn * dw[1], sn * dw[0] + c * dw[1]])
    return D


def ang(a, b):
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = []
    for d in metrics.load():
        mn = d["main"]
        yaw0 = float(mn["base"][0, 2])
        for p in d["pts"]:
            t = p["t"]
            if t + S + 1 > mn["T"]:
                continue
            dm = disp(mn["base"][t:t + S + 1], S)
            dyaw_ep = float(np.degrees((mn["base"][t, 2] - yaw0 + np.pi) % (2 * np.pi) - np.pi))
            m0 = [r for r in p["rolls"] if r["m"] == 0 and r["T"] > S]
            if not m0:
                continue
            Dc = np.array([disp(r["base"], S) for r in m0])
            mean_c = Dc.mean(0)
            # leave-one-out null: each candidate against the mean of the others
            loo = [ang(Dc[i], np.delete(Dc, i, 0).mean(0)) for i in range(len(Dc)) if np.linalg.norm(Dc[i]) > 0.03]
            rows.append(dict(task=d["task"], tag=d["tag"], t=t, k=p["k"], dir=p["dir"].split("/")[0], role=p["role"], hidden=p["hidden"],
                             dyaw_ep=round(dyaw_ep, 1), main_d=float(np.linalg.norm(dm)), cand_d=float(np.linalg.norm(mean_c)),
                             ang_main=ang(dm, mean_c) if np.linalg.norm(dm) > 0.03 and np.linalg.norm(mean_c) > 0.03 else None,
                             ang_loo_med=float(np.median(np.abs(loo))) if loo else None,
                             mode0=float(np.mean([r["a_control_mode"][0, 0, 0] > 0.5 for r in m0]))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, f"base_first{S}.csv.gz"), index=False)
    v = df.dropna(subset=["ang_main"]).copy()
    v["dy"] = v.dyaw_ep.abs()
    v["dyaw_bin"] = pd.cut(v.dy, [-1, 5, 30, 90, 180])
    v["role2"] = v.role.where(~v.role.isin(["cur", "dense", "dense2", "confirm"]), "current")
    g = v.groupby(["task", "role2", "dyaw_bin"], observed=True)
    print(g.agg(points=("ang_main", "size"), med_abs_ang=("ang_main", lambda x: x.abs().median()),
                frac_gt90=("ang_main", lambda x: (x.abs() > 90).mean()), loo_med=("ang_loo_med", "median"),
                med_signed_minus_dyaw=("ang_main", "median")).round(2).to_string())
    # does the main-vs-candidate angle follow the turned angle?
    for role2 in ("current", "sup", "dense_bug3", "bug1"):
        w = v[v.role2 == role2]
        if len(w) > 10:
            c1 = np.corrcoef(w.ang_main, w.dyaw_ep)[0, 1]
            print(f"{role2:10s} n={len(w)} corr(angle main vs candidates, dyaw_ep) = {c1:+.2f}")


if __name__ == "__main__":
    main()
