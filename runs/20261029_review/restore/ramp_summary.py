"""Summary of out/ramp_restore.jsonl: per boundary and per rule, the state error (vs the continuous run, same actions,
16 steps) caused by restoring with settle (v4) + that rule's gripper ramp; 'true' = settle + the true ramp (the rest of
settle's error), 'exact' = the captured true hidden state.
usage: python -I ramp_summary.py"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = ["a_v3", "b_v4", "c_v5", "d_val", "f_hyb", "g_final"]


def g_final_values():
    """g_final (settle_candidate.py) at every scanned boundary, from out/grip_table.npz + out/grip_geo.json."""
    Zf = np.load(f"{HERE}/out/grip_table.npz", allow_pickle=True)
    Z = {k: Zf[k] for k in ("key", "hist", "held", "fix", "obj")}  # NpzFile re-reads an array on every access
    geo = json.load(open(f"{HERE}/out/grip_geo.json"))
    out = {}
    for i, k in enumerate(Z["key"]):
        h = Z["hist"][i]; g0, v1 = h[-1], h[-1] - h[-2]
        if Z["held"][i]:
            v = 1.0 if g0 < 0.0785 else -1.0
        else:
            v = 1.0 if (g0 < 0.07517 and not v1 > 0.0005) else -1.0
            if Z["fix"][i] and not Z["obj"][i] and 0.005 <= g0 < 0.07517 and k in geo and geo[k][0] != geo[k][1]:
                v = 1.0 if geo[k][0] > geo[k][1] else -1.0
        out[k] = v
    return out


def main():
    rows = []
    checks = []
    GF = g_final_values()
    for l in open(f"{HERE}/out/ramp_restore.jsonl"):
        r = json.loads(l)
        checks.append((r["case"]["tag"], r["ramp_check"], r["cont_vs_saved"]))
        for br in r["boundaries"]:
            runs = br["runs"]
            def get(v):
                k = f"ramp={v:+.1f}"
                return runs.get(k)
            ct = round(br["c_true"], 1)
            ent = dict(case=f"{r['case']['task'][:3]} {r['case']['tag'].split('_')[1]} t{r['case']['t']} c{r['case']['c']} b{br['b']}",
                       label=r["case"]["label"].split(";")[0], gap=br["gap_mm"], c_true=ct, held=br["held"], obj=br["obj"])
            key = f"{r['case']['dir']}|{r['case']['tag']}|{r['case']['t']}|{r['case']['c']}|{br['b']}"
            br["rules"]["g_final"] = GF.get(key)
            for name, v in [("true", ct)] + [(rn, br["rules"][rn]) for rn in RULES if br["rules"].get(rn) is not None]:
                x = get(v)
                if x is None:
                    continue
                d16 = x["diff"].get("16") or x["diff"][max(x["diff"], key=int)]
                ent[name] = dict(c=v, gap_dev_mm=x["finger_gap_dev_mm"], obj_dev_m=max((x.get("obj_dev_m") or {}).values(), default=0.0),
                                 arm_q16=d16.get("arm_q", 0.0), finger_q16=d16.get("finger_q", 0.0))
            ent["exact_gap_dev"] = runs["exact"]["finger_gap_dev_mm"]
            rows.append(ent)
    print("ramp check (offline formula vs env current_action at every chunk boundary):")
    for tag, rc, cv in checks:
        print(f"  {tag}: n={rc['n']} max|diff|={rc['max_abs_diff']:.2g} mid-ramp={rc['n_mid']}  (continuous replay vs saved: {cv:.2g})")
    print("\nper boundary: finger-gap deviation over 16 steps (mm) / max object deviation (m), restored with settle + ramp of each rule")
    print(f"{'boundary':34s} {'situation':36s} {'gap':>5s} {'c':>4s} " + " ".join(f"{n:>17s}" for n in ["true"] + RULES))
    agg = {n: [] for n in ["true"] + RULES}
    for e in rows:
        cells = []
        for n in ["true"] + RULES:
            x = e.get(n)
            if x is None:
                cells.append(f"{'-':>17s}"); continue
            agg[n].append((x["gap_dev_mm"], x["obj_dev_m"]))
            cells.append(f"{x['c']:+.1f}:{x['gap_dev_mm']:5.1f}/{x['obj_dev_m']:.3f}")
        print(f"{e['case'][:34]:34s} {e['label'][-36:]:36s} {e['gap']:5.1f} {e['c_true']:+.1f} " + " ".join(f"{c:>17s}" for c in cells))
    print("\nmean (max) over boundaries: finger-gap deviation mm | object deviation m")
    for n, v in agg.items():
        if v:
            a = np.array(v)
            print(f"  {n:6s} {a[:, 0].mean():6.2f} ({a[:, 0].max():5.1f}) | {a[:, 1].mean():.4f} ({a[:, 1].max():.3f})   n={len(a)}")


if __name__ == "__main__":
    main()
