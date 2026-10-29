"""Old row vs settled-v5 row of every re-collected point (same seeds, so the difference is the restore): restored ramp,
how far the main run's finger gap lies outside the 8 candidates' range in the first chunk (steps 1, 2, 4, 8, 16), and
the stage-finish rate V (all rollouts) old vs new. Writes out/paired_v5.csv and prints summaries by reason
(t0 / ramp / yaw, from points_v5.json), task and dataset. usage: python -I paired_v5.py"""
import glob, json, os
import numpy as np
import pandas as pd

D = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"
MAIN = "/home/michaellee/mclee/robocasa/runs/20261025_seen480/run"
HERE = os.path.dirname(os.path.abspath(__file__))
F1, F2 = 12, 13  # finger joints in the flattened state
STEPS = (1, 2, 4, 8, 16)


def rows(d):
    out = {}
    for p in sorted(glob.glob(f"{D}/{d}/results_w*.jsonl")):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                out[(r["tag"], r["t"])] = r
    return out


def outside(d, r, gm):
    G = []
    for i in range(r["N"]):
        s = np.load(f"{D}/{d}/roll/{r['tag']}_t{r['t']}_c{i}_m0.npz")["states"][:17]
        G.append(np.abs(s[:, F1] - s[:, F2]))
    n = min(min(len(g) for g in G), len(gm))
    G = np.array([g[:n] for g in G])
    return {s: 1000 * max(gm[s] - G[:, s].max(), G[:, s].min() - gm[s], 0) for s in STEPS if s < n}


def main():
    why = {(p["out"].split("/")[-1], p["tag"], p["t"]): p for p in json.load(open(f"{HERE}/points_v5.json"))}
    recs, cache = [], {}
    for nd in sorted(glob.glob(f"{D}/run_*_fix5")):
        nd = os.path.basename(nd)
        for (tag, t), r in rows(nd).items():
            od = r.get("replaces") or why[(nd, tag, t)]["dir"]
            if od not in cache:
                cache[od] = rows(od)
            o = cache[od].get((tag, t))
            ep = np.load(f"{MAIN}/{r['task']}/{tag}_steps.npz")["states"]
            gm = np.abs(ep[t:t + 17, F1] - ep[t:t + 17, F2])
            w = why.get((nd, tag, t), {})
            rec = dict(task=r["task"], dataset=w.get("dataset"), dir_new=nd, dir_old=od, tag=tag, t=t, k=r["k"],
                       why="+".join(w.get("why", [])), ramp_old=w.get("old"), ramp_new=r["restore"]["ramp"],
                       held=r["restore"]["held"], fixture=r["restore"]["fixture"],
                       V_new=np.mean([x["done"] for c in r["cands"] for x in c["rollouts"]]))
            for s, v in outside(nd, r, gm).items():
                rec[f"out{s}_new"] = round(v, 2)
            if o is not None:
                rec["V_old"] = np.mean([x["done"] for c in o["cands"] for x in c["rollouts"]])
                for s, v in outside(od, o, gm).items():
                    rec[f"out{s}_old"] = round(v, 2)
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(f"{HERE}/out/paired_v5.csv", index=False)
    df["bad_old"] = df[["out1_old", "out2_old", "out4_old"]].max(axis=1) > 3
    df["bad_new"] = df[["out1_new", "out2_new", "out4_new"]].max(axis=1) > 3
    df["dV"] = df["V_new"] - df["V_old"]
    df["adV"] = df["dV"].abs()
    df["reason"] = np.where(df["why"].str.contains("t0"), "t0", np.where(df["why"].str.contains("ramp"), "ramp", "yaw only"))
    print(f"{len(df)} re-collected points with both rows: {df['V_old'].notna().sum()}")
    for key in ("reason", "task", "dataset"):
        g = df.groupby(key)
        print(f"\nby {key}:")
        print(pd.DataFrame({"n": g.size(), "main outside >3mm in 4 steps: old": g["bad_old"].mean().round(2),
                            "new": g["bad_new"].mean().round(2), "out4 mm old (median)": g["out4_old"].median().round(1),
                            "new ": g["out4_new"].median().round(1), "mean |dV|": g["adV"].mean().round(3),
                            "mean dV": g["dV"].mean().round(3), "|dV| >= 0.25": g["adV"].apply(lambda x: (x >= 0.25).mean()).round(2)}).to_string())
    big = df[df["dV"].abs() >= 0.25].sort_values("dV")
    print(f"\npoints whose V moved by >= 0.25 ({len(big)}):")
    for _, x in big.iterrows():
        print(f"  {x['task'][:3]} {x['dataset']:6s} {x['tag'].split('_')[1]:>4s} t{x['t']:<5d} k{x['k']} {x['why']:10s} ramp {x['ramp_old']:+.1f}->{x['ramp_new']:+.1f}"
              f" V {x['V_old']:.2f}->{x['V_new']:.2f}  out4 {x.get('out4_old', np.nan):.1f}->{x.get('out4_new', np.nan):.1f} mm")


if __name__ == "__main__":
    main()
