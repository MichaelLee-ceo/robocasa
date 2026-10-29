"""settle_v5 vs settle_v4 against the continuous run, per boundary (out/cvr_<task>.jsonl from cont_vs_restore.py)."""
import glob, json
import numpy as np

KEYS = ("arm_q", "finger_q", "obj_q", "base_q")
tot = {"settle_v5": [], "settle_v4": []}
for f in sorted(glob.glob("out/cvr_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        cs = r["case"]
        for br in r["boundaries"]:
            cells = []
            for v in ("exact", "settle_v5", "settle_v4"):
                rr = br["runs"].get(v)
                if rr is None:
                    continue
                vc = rr.get("vs_cont") or rr.get("vs_saved") or {}
                m16 = vc.get("16") or vc.get(16) or {}
                m1 = vc.get("1") or vc.get(1) or {}
                worst16 = max([m16.get(k, 0) for k in KEYS] + [0])
                cells.append(f"{v.replace('settle_', '')}: ramp {rr['ramp'][1]:+.1f} arm16 {m16.get('arm_q', 0) * 1000:5.1f} fing16 {m16.get('finger_q', 0) * 1000:5.1f} obj16 {m16.get('obj_q', 0) * 1000:5.1f}")
                if v in tot:
                    tot[v].append(worst16)
            held = br.get("held", {})
            hd = {k: max((x["max_xyz_diff_m"] for x in v.values()), default=0) for k, v in held.items()} if held else {}
            print(f"{cs['task'][:3]} {cs['tag'].split('_')[1]:>4s} t{cs['t']:<5d} b{br['b']:<4d} | " + " | ".join(cells) + (f" | held obj max dev (mm) {({k.replace('settle_', ''): round(v * 1000, 1) for k, v in hd.items()})}" if hd else ""))
for v, x in tot.items():
    x = np.array(x) * 1000
    if len(x):
        print(f"{v}: n {len(x)}, worst-group diff after 16 steps (mm/mrad): median {np.median(x):.2f}, mean {x.mean():.2f}, max {x.max():.1f}, > 5: {(x > 5).sum()}")
