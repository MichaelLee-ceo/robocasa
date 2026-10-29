"""Builds out/grip_table.npz: one row per scanned chunk boundary (out/boundaries_<task>.jsonl), with the finger gap over
the last 12 steps, the exact ramp c (start ramp of the row's settle version + 0.2 per saved command), the last and next
commands, and the contact flags of the scan (held = both fingers on one free object; object = any finger on a free object;
fixture = any finger on a fixture). Rules are then evaluated vectorised (grip_eval.py).
usage: python -I grip_table.py"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_scan import infer, rule_of, FQ, D, R, OUT, rows, DIRS  # noqa: E402
from grip_rules3 import FIXTURE_WORDS  # noqa: E402

TASKS = list(DIRS)


def main():
    rowmap = {}
    for task, dirs in DIRS.items():
        for d in dirs:
            for r in rows(d):
                rowmap[(d, r["tag"], r["t"])] = r
    cols = {k: [] for k in ("task", "hist", "c", "cmd_last", "cmd_next", "held", "obj", "fix", "k", "yaw", "key")}
    for ti, task in enumerate(TASKS):
        recs = [json.loads(l) for l in open(f"{OUT}/boundaries_{task}.jsonl")]
        recs.sort(key=lambda r: (r["dir"], r["tag"], r["t"], r["c"], r["b"]))
        cur, ep_tag = None, None
        for rec in recs:
            key = (rec["dir"], rec["tag"], rec["t"], rec["c"])
            if key != cur:
                cur = key
                row = rowmap[(rec["dir"], rec["tag"], rec["t"])]
                if ep_tag != rec["tag"]:
                    ep = np.load(f"{R}/20261025_seen480/run/{task}/{rec['tag']}_steps.npz")["states"]; ep_tag = rec["tag"]
                t = rec["t"]
                c0 = infer(ep[t], ep[t - 1] if t else None, rule_of(row.get("hidden")))
                z = np.load(f"{D}/{rec['dir']}/roll/{rec['tag']}_t{t}_c{rec['c']}_m0.npz")
                S, A = z["states"], z["action.gripper_close"][:, :, 0].reshape(-1)
                cmd = np.where(A < 0.5, -1.0, 1.0)
                c = [c0]
                for a in cmd[: len(S) - 1]:
                    c.append(float(np.clip(c[-1] + 0.2 * a, -1, 1)))
                g = np.abs(S[:, FQ[0]] - S[:, FQ[1]])
            b = rec["b"]
            cols["task"].append(ti)
            cols["hist"].append(g[b - 12:b + 1] if b >= 12 else np.concatenate([np.full(12 - b, g[0]), g[:b + 1]]))
            cols["c"].append(c[b]); cols["cmd_last"].append(cmd[b - 1]); cols["cmd_next"].append(cmd[b] if b < len(cmd) else 0)
            cols["held"].append(bool(rec["held"]))
            cols["obj"].append(any(o.endswith("_main") and not any(w in o for w in FIXTURE_WORDS) for o in rec["touch"]))
            cols["fix"].append(any(any(w in o for w in FIXTURE_WORDS) for o in rec["touch"]))
            cols["k"].append(rec["k"]); cols["yaw"].append(rec["yaw_deg"])
            cols["key"].append(f"{rec['dir']}|{rec['tag']}|{rec['t']}|{rec['c']}|{b}")
        print(task, len(recs), flush=True)
    np.savez_compressed(f"{OUT}/grip_table.npz", **{k: np.array(v) for k, v in cols.items()}, tasks=np.array(TASKS))


if __name__ == "__main__":
    main()
