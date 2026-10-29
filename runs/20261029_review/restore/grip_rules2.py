"""Accuracy of settle's gripper-side inference (coordinator request 2026-10-29), against the exact ramp.

Truth. PandaGripper.current_action = [-c, c]; c moves by 0.2 per policy step toward the command (+1 close, -1 open) and
is exact given the ramp a rollout started with (settle's inference at its decision point, with the rule of the row's
version) and its saved actions. (Checked against env.robots[0].gripper["right"].current_action read in continuous
replays: see out/ramp_check_*.json.) Every chunk boundary b = 16k (k >= 1) of every saved rollout of the current-code
dirs is a candidate restore point.

Rules (+1 closing, -1 opening; gap = finger gap, gN = gap N steps earlier, from the saved states):
  a_v3  : closing iff gap < 75.17 mm (1.9 x init gap - 4 mm) and not gap - g1 > 0.5 mm   (rows up to settled-v3)
  b_v4  : closing iff gap < 78.5 mm (joint-limit gap - 1.5 mm) and not gap - g1 > 0.5 mm (hidden_state.py now)
  c_v5  : d = gap - g8; closing if d < -0.3 mm, opening if d > +0.3 mm, else closing iff gap < 78.5 mm (candidate)
  plus variants of c_v5 (window 4 / 12 steps, +-0.1 mm) and d_t10 (last 10 steps: widened > 0.3 mm from their minimum ->
  opening; narrowed > 0.3 mm from their maximum -> closing; else closing iff gap < 77 mm).

Scores. Saturated boundaries (|c| = 1): wrong side or not. Mid-ramp boundaries (|c| < 1): settle can only set +-1, so the
error is |c_rule - c| / 0.2 = the number of policy steps the restored ramp is ahead of / behind the true one (also
reported: the unavoidable part, min over +-1). Error in steps is reported over all boundaries too (a wrong side = 10).
Situations come from out/boundaries_<task>.jsonl (contacts, held objects; every 2nd candidate's m0 rollout).
usage: python -I grip_rules2.py -> out/grip_rules2.json, out/grip_rules2.txt"""
import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_scan import DIRS, rows, infer, rule_of, FQ, D, R, OUT  # noqa: E402

OPEN_V4, OPEN_V3 = 0.0785, abs(0.020833 * 2) * 1.9 - 0.004


def rule(name, g, b):
    x = g[b]
    if name == "a_v3":
        return -1.0 if (x >= OPEN_V3 or x - g[b - 1] > 0.0005) else 1.0
    if name == "b_v4":
        return -1.0 if (x >= OPEN_V4 or x - g[b - 1] > 0.0005) else 1.0
    if name.startswith("c_v5"):
        w, th = {"c_v5": (8, 0.0003), "c_v5_w4": (4, 0.0003), "c_v5_w12": (12, 0.0003), "c_v5_th01": (8, 0.0001)}[name]
        dd = x - g[max(b - w, 0)]
        if dd < -th:
            return 1.0
        if dd > th:
            return -1.0
        return 1.0 if x < OPEN_V4 else -1.0
    if name == "d_t10":
        w = g[max(b - 10, 0):b + 1]
        if x - w.min() > 0.0003:
            return -1.0
        if w.max() - x > 0.0003:
            return 1.0
        return 1.0 if x < 0.077 else -1.0
    raise KeyError(name)


RULES = ["a_v3", "b_v4", "c_v5", "c_v5_w4", "c_v5_w12", "c_v5_th01", "d_t10"]


def situation(rec, g, b, c, cmd):
    """coarse situation label of a scanned boundary."""
    gap = g[b] * 1000
    if abs(c[b]) < 0.999:
        return "mid-ramp: grasping (last cmd close)" if cmd[b - 1] > 0 else "mid-ramp: releasing (last cmd open)"
    if rec is None:
        return None
    held = rec["held"]
    objs = [o for o in rec["touch"] if o.endswith("_main") and o[:-5] in ("obj", "bowl", "vegetable") or o.startswith(("meat", "veg", "fruit", "obj"))]
    if held:
        return "holding, gap >= 60 mm (wide grasp)" if gap >= 60 else ("holding, gap 20-60 mm" if gap >= 20 else "holding, gap < 20 mm")
    if rec["touch"]:
        if any(o in rec["touch"] for o in [x for x in rec["touch"] if x.endswith("_main") and not any(
                f in x for f in ("counter", "stove", "microwave", "fridge", "sink", "cab", "stack", "wall", "floor"))]):
            return "touching an object (one finger / not held)"
        return "touching a fixture"
    if gap >= 78.5:
        return "free, fully open (>= 78.5 mm)"
    if gap >= 75.17:
        return "free, 75.2-78.5 mm band (creep / slow open)"
    if gap < 5:
        return "free, closed (< 5 mm)"
    return "free, 5-75 mm"


def main():
    scanned = {}
    for task in DIRS:
        p = f"{OUT}/boundaries_{task}.jsonl"
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l)
                scanned[(r["dir"], r["tag"], r["t"], r["c"], r["b"])] = r
    agg = defaultdict(lambda: defaultdict(lambda: dict(n=0, wrong=0, err_steps=0.0)))  # (task, situation) -> rule -> stats
    tot = defaultdict(lambda: dict(n=0, sat=0, mid=0, mid_quant=0.0))
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
                    m = cand["rollouts"][0]["m"]
                    f = f"{D}/{d}/roll/{tag}_t{t}_c{ci}_m{m}.npz"
                    if not os.path.exists(f):
                        continue
                    z = np.load(f)
                    S, A = z["states"], z["action.gripper_close"][:, :, 0].reshape(-1)
                    cmd = np.where(A < 0.5, -1.0, 1.0)
                    c = [c0]
                    for a in cmd[: len(S) - 1]:
                        c.append(float(np.clip(c[-1] + 0.2 * a, -1, 1)))
                    g = np.abs(S[:, FQ[0]] - S[:, FQ[1]])
                    # the history before the rollout's start comes from the main run
                    gh = np.concatenate([np.abs(ep[max(t - 12, 0):t, FQ[0]] - ep[max(t - 12, 0):t, FQ[1]]), g])
                    off = len(gh) - len(g)
                    for b in range(16, len(S), 16):
                        ct = c[b]
                        sat = abs(ct) > 0.999
                        T = tot[task]
                        T["n"] += 1
                        if sat:
                            T["sat"] += 1
                        else:
                            T["mid"] += 1; T["mid_quant"] += min(abs(1 - ct), abs(-1 - ct)) / 0.2
                        rec = scanned.get((d, tag, t, ci, b)) if m == 0 else None
                        sit = situation(rec, g, b, c, cmd)
                        keys = [(task, "ALL boundaries"), (task, "saturated" if sat else "mid-ramp")]
                        if sit:
                            keys.append((task, sit))
                        for rn in RULES:
                            v = rule(rn, gh, b + off)
                            err = abs(v - ct) / 0.2
                            for k in keys:
                                s = agg[k][rn]
                                s["n"] += 1; s["err_steps"] += err
                                if sat and v != np.sign(ct):
                                    s["wrong"] += 1
            print(task, d, dict(tot[task]), flush=True)
    # known main-run failure cases (truth from where the main run's fingers go next)
    known = {}
    for task, tag, t in [("KettleBoiling", "KettleBoiling_s26_single", 112), ("PackIdenticalLunches", "PackIdenticalLunches_s14_single", 16)]:
        ep = np.load(f"{R}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
        g = np.abs(ep[:, FQ[0]] - ep[:, FQ[1]])
        known[f"{tag} t{t}"] = dict(gap_mm_t_minus8_to_plus14=[round(x * 1000, 1) for x in g[t - 8:t + 15]],
                                    rules={rn: rule(rn, g, t) for rn in RULES})
    lines = []
    for task in DIRS:
        T = tot[task]
        lines.append(f"\n== {task}: {T['n']} boundaries, {T['sat']} saturated, {T['mid']} mid-ramp ({T['mid'] / max(T['n'], 1):.1%}); "
                     f"unavoidable +-1 error at mid-ramp boundaries: {T['mid_quant'] / max(T['mid'], 1):.2f} steps on average")
        lines.append(f"{'situation':48s} {'n':>6s} " + " ".join(f"{rn:>16s}" for rn in RULES))
        sits = sorted({k[1] for k in agg if k[0] == task}, key=lambda s: (s != "ALL boundaries", s != "saturated", s != "mid-ramp", s))
        for s in sits:
            st = agg[(task, s)]
            n = st[RULES[0]]["n"]
            cells = []
            for rn in RULES:
                x = st[rn]
                if s.startswith("mid-ramp"):
                    cells.append(f"{x['err_steps'] / max(x['n'], 1):9.2f} steps")
                else:
                    cells.append(f"{x['wrong'] / max(x['n'], 1):6.2%} {x['err_steps'] / max(x['n'], 1):5.2f}st")
            lines.append(f"{s[:48]:48s} {n:6d} " + " ".join(f"{cc:>16s}" for cc in cells))
    lines.append("\nknown main-run cases (+1 closing, -1 opening):")
    for k, v in known.items():
        lines.append(f"  {k}: {v['rules']}\n     gap t-8..t+14 (mm): {v['gap_mm_t_minus8_to_plus14']}")
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/grip_rules2.txt", "w").write(txt + "\n")
    json.dump(dict(tot=tot, agg={f"{k[0]}|{k[1]}": v for k, v in agg.items()}, known=known), open(f"{OUT}/grip_rules2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
