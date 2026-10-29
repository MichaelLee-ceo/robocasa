import json, sys
import numpy as np
R_DIR = "/home/michaellee/mclee/robocasa/runs/20261029_review/restore"
sys.path.insert(0, R_DIR)
import harness as H
grid = lambda x: float(np.clip(np.round(x / 0.2) * 0.2, -1, 1))
OPEN_V3 = abs(0.020833 * 2) * 1.9 - 0.004
rows = [json.loads(l) for l in open(f"{R_DIR}/out/ramp_restore.jsonl")]
tot = {}
for r in rows:
    cs = r["case"]
    ss = np.load(H.roll_path(cs["dir"], cs["tag"], cs["t"], cs["c"], cs["m"]))["states"]
    g = np.abs(ss[:, 12] - ss[:, 13])
    for br in r["boundaries"]:
        b = br["b"]; g0, g1 = g[b], g[b - 1]; v1 = g0 - g1
        free = not br["held"] and not br["obj"]
        runs = br["runs"]
        vals = {k: v["finger_gap_dev_mm"] for k, v in runs.items()}
        od = {k: max(v.get("obj_dev_m", {}).values(), default=0) for k, v in runs.items()}
        hyb = {}
        for lag in (1, 2):
            if free and 0.005 <= g0 < OPEN_V3:
                hyb[lag] = grid(1 - g0 / 0.04 + lag * 0.2 * np.sign(-v1) * (abs(v1) > 0.0003))
            else:
                hyb[lag] = None
        key = lambda v: f"ramp={v:+.1f}"
        print(f"{cs['task'][:3]} {cs['tag'].split('_')[1]} t{cs['t']} c{cs['c']} b{b}: gap {g0*1000:5.1f} v1 {v1*1000:+5.2f} held {br['held']:d} obj {br['obj']:d} c_true {br['c_true']:+.1f}"
              f" | b_v4 {br['rules']['b_v4']:+.0f}->{vals.get(key(br['rules']['b_v4']))} | a_v3 {br['rules']['a_v3']:+.0f}->{vals.get(key(br['rules']['a_v3']))}"
              f" | true->{vals.get(key(br['c_true']))} | hyb1 {hyb[1]} -> {vals.get(key(hyb[1])) if hyb[1] is not None else '-'} hyb2 {hyb[2]} -> {vals.get(key(hyb[2])) if hyb[2] is not None else '-'}"
              f" | tested {sorted(vals)}")
