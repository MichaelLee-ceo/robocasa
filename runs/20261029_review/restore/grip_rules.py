"""Offline evaluation of gripper-ramp inference rules against the exact ramp at every chunk boundary of the saved
rollouts (same data and truth as grip_scan.py). A rule returns +1 (closing) or -1 (opening); errors are counted where
the true ramp is saturated (|c| = 1) and the rule picks the other side ("wrong side"), split by the true side.
Rules:
  v3, v4: settle's rules (1-step widening > 0.5 mm, open thresholds 75.17 / 78.5 mm)
  v4_w5: v4 threshold, widening judged over 5 steps (> 0.1 mm)
  trend: opening iff the gap widened by > 0.1 mm over the last 5 steps or gap >= 79 mm; otherwise closing
  trend10: last 10 steps: widened by > 0.3 mm from their minimum -> opening; narrowed by > 0.3 mm from their maximum
           -> closing; static: gap >= 77 mm -> opening, else closing
usage: python -I grip_rules.py -> out/grip_rules.json"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_scan import DIRS, rows, infer, rule_of, FQ, D, R, OUT  # noqa: E402


def gaps(S):
    return np.abs(S[:, FQ[0]] - S[:, FQ[1]])


def rule(name, g, b):
    x = g[b]
    if name in ("v3", "v4"):
        thr = 0.0785 if name == "v4" else abs(0.020833 * 2) * 1.9 - 0.004
        return -1.0 if (x >= thr or x - g[b - 1] > 0.0005) else 1.0
    if name == "v4_w5":
        return -1.0 if (x >= 0.0785 or x - g[max(b - 5, 0)] > 0.0001) else 1.0
    if name == "trend":
        return -1.0 if (x >= 0.079 or x - g[max(b - 5, 0)] > 0.0001) else 1.0
    if name == "trend10":
        w = g[max(b - 10, 0):b + 1]
        if x - w.min() > 0.0003:
            return -1.0
        if w.max() - x > 0.0003:
            return 1.0
        return -1.0 if x >= 0.077 else 1.0
    raise KeyError(name)


RULES = ["v3", "v4", "v4_w5", "trend", "trend10"]


def main():
    res = defaultdict(Counter)
    for task, dirs in DIRS.items():
        ep_cache = {}
        for d in dirs:
            for r in rows(d):
                tag, t = r["tag"], r["t"]
                if tag not in ep_cache:
                    ep_cache = {tag: np.load(f"{R}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]}
                ep = ep_cache[tag]
                c0 = infer(ep[t], ep[t - 1] if t else None, rule_of(r.get("hidden")))
                for ci, cand in enumerate(r["cands"]):
                    ro = cand["rollouts"][0]
                    f = f"{D}/{d}/roll/{tag}_t{t}_c{ci}_m{ro['m']}.npz"
                    if not os.path.exists(f):
                        continue
                    z = np.load(f)
                    S, A = z["states"], z["action.gripper_close"][:, :, 0].reshape(-1)
                    cmd = np.where(A < 0.5, -1.0, 1.0)
                    c = [c0]
                    for a in cmd[: len(S) - 1]:
                        c.append(float(np.clip(c[-1] + 0.2 * a, -1, 1)))
                    g = gaps(S)
                    for b in range(16, len(S) - 16, 16):
                        ct = c[b]
                        if abs(ct) < 0.999:
                            res[(task, "all")]["mid_ramp"] += 1
                            continue
                        res[(task, "all")]["saturated"] += 1
                        for rn in RULES:
                            v = rule(rn, g, b)
                            if v != np.sign(ct):
                                res[(task, rn)]["open_read_closing" if ct < 0 else "closed_read_opening"] += 1
            print(task, d, {k[1]: dict(v) for k, v in res.items() if k[0] == task}, flush=True)
    out = {f"{k[0]}|{k[1]}": dict(v) for k, v in res.items()}
    json.dump(out, open(f"{OUT}/grip_rules.json", "w"), indent=1)
    for task in DIRS:
        n = res[(task, "all")]["saturated"]
        print(task, "saturated boundaries", n, "mid-ramp", res[(task, "all")]["mid_ramp"])
        for rn in RULES:
            v = res[(task, rn)]
            print(f"   {rn:8s} open read as closing {v['open_read_closing']:5d} ({v['open_read_closing'] / n:.2%})   "
                  f"closed read as opening {v['closed_read_opening']:5d} ({v['closed_read_opening'] / n:.2%})")


if __name__ == "__main__":
    main()
