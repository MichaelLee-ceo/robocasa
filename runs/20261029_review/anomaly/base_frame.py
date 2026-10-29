"""Base-command frame check, without the main run's actions: in base mode the policy's base command (base_motion[0:2],
in the robot's current frame) should move the base in that direction. For each rollout, over the base-mode steps of
its first two chunks, the summed command and the realized base displacement (world, rotated into the robot frame at
each step) are compared: angle(command, motion). With a correct controller frame this is ~0 whatever the base turned
since the episode start (dyaw_ep); with bug 3 it equals dyaw_ep. Also reports the main run's base motion at the same
point for context. Writes base_frame.csv.gz. usage: python -I base_frame.py"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402


def ang(a, b):
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def main():
    rows = []
    for d in metrics.load():
        mn = d["main"]
        yaw0 = float(mn["base"][0, 2])
        for p in d["pts"]:
            t = p["t"]
            dyaw_ep = float(np.degrees((mn["base"][t, 2] - yaw0 + np.pi) % (2 * np.pi) - np.pi))
            for r in p["rolls"]:
                if r["m"] != 0:
                    continue
                n = r["T"] - 1
                mode = r["a_control_mode"][:, :, 0].reshape(-1)[:n] > 0.5
                cmd = r["a_base_motion"][:, :, 0:2].reshape(-1, 2)[:n]
                b = r["base"]
                C, D, nb = np.zeros(2), np.zeros(2), 0
                for s in range(min(n, 32)):
                    if not mode[s]:
                        continue
                    dw = b[s + 1, :2] - b[s, :2]
                    c, sn = np.cos(-b[s, 2]), np.sin(-b[s, 2])
                    D += np.array([c * dw[0] - sn * dw[1], sn * dw[0] + c * dw[1]])
                    C += cmd[s]
                    nb += 1
                rows.append(dict(task=d["task"], tag=d["tag"], t=t, k=p["k"], dir=p["dir"].split("/")[0], role=p["role"], hidden=p["hidden"],
                                 c=r["c"], dyaw_ep=round(dyaw_ep, 1), n_base=nb, cmd=np.linalg.norm(C) / max(nb, 1),
                                 disp=float(np.linalg.norm(D)), ang=ang(C, D) if nb >= 4 and np.linalg.norm(D) > 0.03 and np.linalg.norm(C) > 0 else None))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "base_frame.csv.gz"), index=False)
    v = df.dropna(subset=["ang"])
    v = v.assign(err=lambda x: x.ang.abs(), dy=lambda x: x.dyaw_ep.abs())
    v["dyaw_bin"] = pd.cut(v.dy, [-1, 5, 30, 90, 180])
    print(v.groupby(["task", "role", "dyaw_bin"], observed=True).agg(n=("err", "size"), med_err=("err", "median"), p90_err=("err", lambda x: x.quantile(0.9)),
                                                                      frac_gt45=("err", lambda x: (x > 45).mean()),
                                                                      corr_with_dyaw=("ang", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
