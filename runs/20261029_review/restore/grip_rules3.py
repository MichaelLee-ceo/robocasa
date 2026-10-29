"""Gripper-ramp inference rules vs the exact ramp (coordinator requests 2026-10-29), on every scanned chunk boundary
(out/boundaries_<task>.jsonl: every 2nd candidate's m0 rollout of the current-code dirs, with finger contacts).

Truth c (PandaGripper.current_action = [-c, c]): the rollout's start ramp (settle's inference at its decision point,
rule of the row's version) + 0.2 per step toward each saved command; verified against the env's current_action in
continuous replays (ramp_restore.py, out/ramp_restore_*.jsonl: ramp_check).

Rules (c in [-1, 1]; +1 = close; g = finger gap, gN = gap N steps earlier, v = g - g1):
  a_v3  closing iff g < 75.17 mm and not v > 0.5 mm                                   (rows up to settled-v3)
  b_v4  closing iff g < 78.5 mm and not v > 0.5 mm                                    (hidden_state.py now)
  c_v5  d8 = g - g8: closing if d8 < -0.3 mm, opening if d8 > +0.3 mm, else closing iff g < 78.5 mm
  d_val ramp VALUE: opening (-1) if g - g4 > 0.2 mm; closing (+1) if g - g4 < -0.2 mm; steady: an object between /
        against the fingers (finger contact with a free object) -> +1, else c = 1 - g / 40 mm rounded to 0.2
  d_valv same, also opening if v > 0.05 mm (finger velocity > 1 mm/s), closing if v < -0.05 mm
  e_v5val c_v5's side, but a steady free mid gap (5-75 mm, no object contact) gets d_val's value
Situations: t=0 handled separately (true c = 0 after reset). Scores: wrong side at saturated boundaries; mean |c_rule - c|
in policy steps (/0.2) per situation, overall, and at mid-ramp boundaries.
usage: python -I grip_rules3.py -> out/grip_rules3.txt / .json"""
import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_scan import infer, rule_of, FQ, D, R, OUT, rows, DIRS  # noqa: E402

OPEN_V4, OPEN_V3 = 0.0785, abs(0.020833 * 2) * 1.9 - 0.004
FIXTURE_WORDS = ("counter", "stove", "microwave", "fridge", "sink", "cab", "stack", "wall", "floor", "coffee", "toaster",
                 "dishwasher", "oven", "hood", "box", "window", "shelf", "drawer", "bottom", "corner", "island", "outlet")
RULES = ["a_v3", "b_v4", "c_v5", "d_val", "d_valv", "e_v5val"]


def object_contact(rec):
    return any(o.endswith("_main") and not any(w in o for w in FIXTURE_WORDS) for o in rec["touch"])


def rule(name, g, b, obj):
    x = g[b]
    v = x - g[b - 1]
    if name == "a_v3":
        return -1.0 if (x >= OPEN_V3 or v > 0.0005) else 1.0
    if name == "b_v4":
        return -1.0 if (x >= OPEN_V4 or v > 0.0005) else 1.0
    if name in ("c_v5", "e_v5val"):
        d8 = x - g[max(b - 8, 0)]
        if d8 < -0.0003:
            return 1.0
        if d8 > 0.0003:
            return -1.0
        if name == "e_v5val" and 0.005 <= x < 0.075 and not obj:
            return float(np.clip(np.round((1 - x / 0.04) / 0.2) * 0.2, -1, 1))
        return 1.0 if x < OPEN_V4 else -1.0
    if name in ("d_val", "d_valv"):
        d4 = x - g[max(b - 4, 0)]
        if d4 > 0.0002 or (name == "d_valv" and v > 0.00005):
            return -1.0
        if d4 < -0.0002 or (name == "d_valv" and v < -0.00005):
            return 1.0
        if obj:
            return 1.0
        return float(np.clip(np.round((1 - x / 0.04) / 0.2) * 0.2, -1, 1))
    raise KeyError(name)


def situation(rec, g, b, c, cmd):
    x = g[b] * 1000
    obj = object_contact(rec)
    if abs(c[b]) < 0.999:
        if 5 <= x < 75 and abs(g[b] - g[b - 4]) * 1000 < 0.2:
            return "mid-ramp, steady mid gap (5-75 mm)"
        return "mid-ramp, releasing (last cmd open)" if cmd[b - 1] < 0 else "mid-ramp, grasping (last cmd close)"
    if rec["held"]:
        return "holding, wide grasp (>= 60 mm)" if x >= 60 else ("holding, 20-60 mm" if x >= 20 else "holding, < 20 mm")
    if obj:
        return "touching an object, not held"
    if rec["touch"]:
        if 5 <= x < 75:
            return "touching a fixture, mid gap 5-75 mm"
        return "touching a fixture, gap < 5 or >= 75 mm"
    if x >= 78.5:
        return "free, >= 78.5 mm"
    if x >= 75.17:
        return "free, 75.2-78.5 mm (creep band)"
    if x < 5:
        return "free, closed < 5 mm"
    return "free, mid gap 5-75 mm"


def main():
    agg = defaultdict(lambda: defaultdict(lambda: dict(n=0, wrong=0, err=0.0, sat=0)))
    rowmap = {}
    for task, dirs in DIRS.items():
        for d in dirs:
            for r in rows(d):
                rowmap[(d, r["tag"], r["t"])] = r
    examples = defaultdict(list)
    for task in DIRS:
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
            ct = c[b]
            sat = abs(ct) > 0.999
            sit = situation(rec, g, b, c, cmd)
            obj = object_contact(rec)
            for k in ((task, "ALL"), ("ALL", "ALL"), (task, sit), ("ALL", sit), (task, "saturated" if sat else "mid-ramp (all)")):
                for rn in RULES:
                    cr = rule(rn, g, b, obj)
                    s = agg[k][rn]
                    s["n"] += 1; s["err"] += abs(cr - ct) / 0.2
                    if sat:
                        s["sat"] += 1
                        if np.sign(cr) != np.sign(ct) or cr == 0:
                            s["wrong"] += 1
            if sit.startswith("mid-ramp, steady") and len(examples[sit]) < 30:
                examples[sit].append(dict(task=task, dir=rec["dir"], tag=rec["tag"], t=rec["t"], c=rec["c"], b=b,
                                          gap_mm=round(g[b] * 1000, 1), c_true=round(ct, 2), touch=sorted(rec["touch"]),
                                          cmds_last10=[int(x) for x in cmd[b - 10:b]]))
    # t = 0: the true ramp after load_scene is exactly 0 (robot.reset zeroes it; read in cont_vs_restore b0)
    t0 = {}
    for task, tag in (("KettleBoiling", "KettleBoiling_s0_single"), ("SteamInMicrowave", "SteamInMicrowave_s0_single"), ("PackIdenticalLunches", "PackIdenticalLunches_s0_single")):
        ep = np.load(f"{R}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
        g = np.abs(ep[:1, FQ[0]] - ep[:1, FQ[1]])
        gg = np.repeat(g, 10)  # no history at t = 0: treated as steady
        t0[task] = {rn: rule(rn, gg, 9, False) for rn in RULES}
    known = {}
    for task, tag, t, obj in [("KettleBoiling", "KettleBoiling_s26_single", 112, True), ("PackIdenticalLunches", "PackIdenticalLunches_s14_single", 16, False)]:
        ep = np.load(f"{R}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
        g = np.abs(ep[:, FQ[0]] - ep[:, FQ[1]])
        known[f"{tag} t{t}"] = dict(gap_mm=[round(x * 1000, 1) for x in g[t - 8:t + 15]], rules={rn: rule(rn, g, t, obj) for rn in RULES})
    lines = []
    order = ["ALL", "saturated", "mid-ramp (all)"]
    for task in ["ALL"] + list(DIRS):
        sits = sorted({k[1] for k in agg if k[0] == task}, key=lambda s: (order.index(s) if s in order else 9, s))
        lines.append(f"\n== {task}: cells = wrong side at saturated boundaries (%) / mean |c_rule - c_true| in policy steps")
        lines.append(f"{'situation':40s} {'n':>6s} {'n_sat':>6s} " + " ".join(f"{rn:>15s}" for rn in RULES))
        for s in sits:
            st = agg[(task, s)]
            n, ns = st[RULES[0]]["n"], st[RULES[0]]["sat"]
            cells = [(f"{st[rn]['wrong'] / ns:6.1%}" if ns else "     -") + f" / {st[rn]['err'] / n:4.2f}" for rn in RULES]
            lines.append(f"{s[:40]:40s} {n:6d} {ns:6d} " + " ".join(f"{cc:>15s}" for cc in cells))
    lines.append("\nt = 0 (true c = 0 after load_scene): rule outputs " + json.dumps(t0))
    lines.append("known main-run cases (c: +1 closing, -1 opening):")
    for k, v in known.items():
        lines.append(f"  {k}: {v['rules']}  gap t-8..t+14 (mm) {v['gap_mm']}")
    lines.append("\nexamples, mid-ramp with a steady mid gap:")
    for s, ex in examples.items():
        for e in ex[:12]:
            lines.append("  " + json.dumps(e))
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/grip_rules3.txt", "w").write(txt + "\n")
    json.dump(dict(agg={f"{k[0]}|{k[1]}": v for k, v in agg.items()}, t0=t0, known=known, examples=examples),
              open(f"{OUT}/grip_rules3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
