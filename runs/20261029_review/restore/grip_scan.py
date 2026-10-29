"""Offline (no sim) check of hidden_state.settle's gripper-ramp inference.

The Panda gripper ramp (PandaGripper.current_action = [-c, c]) moves c by 0.2 per policy step toward the commanded side
(+1 close, -1 open) and is exactly computable from the saved actions. For every chunk boundary inside every saved rollout
(the states a decision point would be placed at), compare the true c with what settle infers from the finger gap:
  v3 rule (rows "settled", "settled-basefix", "settled-v3"): closing if gap < 1.9 * init gap - 4 mm (75.17 mm)
  v4 rule (current code):                                  closing if gap < joint-limit gap - 1.5 mm (78.5 mm)
  both: unless the gap widened by > 0.5 mm in the last step.
The rollout's own start ramp is settle's inference at the decision point (rule of the row's version), so c is exact
along the rollout. Also scans the decision points themselves (main-run states; true command unknown there).
usage: python -I grip_scan.py  -> out/grip_scan.json"""
import glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np

R = "/home/michaellee/mclee/robocasa/runs"
D = f"{R}/20261026_critic_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FQ = (12, 13)  # flattened-state index of the two finger joints (qpos 11, 12): same in every scene of the 3 tasks
OPEN_V4 = 0.08 - 0.0015
OPEN_V3 = abs(0.020833 - (-0.020833)) * 1.9 - 0.004


def gap(s):
    return abs(s[FQ[0]] - s[FQ[1]])


def infer(s, prev, rule):
    g = gap(s)
    closing = g < (OPEN_V4 if rule == "v4" else OPEN_V3)
    if prev is not None and closing:
        closing = not (g - gap(prev) > 0.0005)
    return 1.0 if closing else -1.0


def rule_of(hidden):
    return "v4" if hidden == "settled-v4" else "v3"


DIRS = {"KettleBoiling": ["run_v2/s0", "run_v2/s1", "run_v2/s2", "run_v2/s3", "run_v2/s4", "run_dense_KettleBoiling",
                          "run_confirm_KettleBoiling"],
        "SteamInMicrowave": ["run_SteamInMicrowave_fix3", "run_dense_SteamInMicrowave", "run_dense2_SteamInMicrowave"],
        "PackIdenticalLunches": ["run_PackIdenticalLunches_fix3", "run_dense_PackIdenticalLunches",
                                 "run_dense2_PackIdenticalLunches"]}


def rows(d):
    out = []
    for p in glob.glob(f"{D}/{d}/results_w*.jsonl"):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                out.append(r)
    return out


def main():
    res = dict(boundaries=defaultdict(Counter), examples=defaultdict(list), points={})
    ep_cache = {}
    for task, dirs in DIRS.items():
        for d in dirs:
            for r in rows(d):
                tag, t = r["tag"], r["t"]
                if tag not in ep_cache:
                    ep_cache = {tag: np.load(f"{R}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]}
                ep = ep_cache[tag]
                c0 = infer(ep[t], ep[t - 1] if t else None, rule_of(r.get("hidden")))
                for ci, cand in enumerate(r["cands"]):
                    for ro in cand["rollouts"]:
                        f = f"{D}/{d}/roll/{tag}_t{t}_c{ci}_m{ro['m']}.npz"
                        if not os.path.exists(f):
                            continue
                        z = np.load(f)
                        S, A = z["states"], z["action.gripper_close"][:, :, 0].reshape(-1)
                        cmd = np.where(A < 0.5, -1.0, 1.0)
                        c = [c0]
                        for a in cmd[: len(S) - 1]:
                            c.append(float(np.clip(c[-1] + 0.2 * a, -1, 1)))
                        for b in range(16, len(S) - 16, 16):
                            ct = c[b]
                            last = cmd[b - 1]
                            for rule in ("v3", "v4"):
                                ci_ = infer(S[b], S[b - 1], rule)
                                if abs(ct) > 0.999:
                                    kind = "exact" if ci_ == np.sign(ct) else "WRONG_SIDE"
                                else:
                                    kind = "mid_ramp_same_dir" if ci_ == last else "mid_ramp_WRONG_DIR"
                                res["boundaries"][f"{task}|{rule}"][kind] += 1
                                if kind in ("WRONG_SIDE", "mid_ramp_WRONG_DIR") and len(res["examples"][f"{task}|{rule}|{kind}"]) < 400:
                                    res["examples"][f"{task}|{rule}|{kind}"].append(
                                        dict(dir=d, tag=tag, t=t, cand=ci, m=ro["m"], b=b, gap_mm=round(gap(S[b]) * 1000, 2),
                                             dgap_mm=round((gap(S[b]) - gap(S[b - 1])) * 1000, 3), c_true=round(ct, 2),
                                             c_settle=ci_, last_cmd=float(last)))
                # the decision point itself (main-run state; true ramp unknown)
                g, gp = gap(ep[t]), (gap(ep[t - 1]) if t else None)
                res["points"][f"{d}|{tag}|{t}"] = dict(task=task, gap_mm=round(g * 1000, 2),
                                                       dgap_mm=None if gp is None else round((g - gp) * 1000, 3),
                                                       dgap5_mm=None if t < 5 else round((g - gap(ep[t - 5])) * 1000, 3),
                                                       v3=infer(ep[t], ep[t - 1] if t else None, "v3"),
                                                       v4=infer(ep[t], ep[t - 1] if t else None, "v4"),
                                                       hidden=r.get("hidden"), k=r["k"])
            print(task, d, dict(res["boundaries"]), flush=True)
    res["boundaries"] = {k: dict(v) for k, v in res["boundaries"].items()}
    json.dump(res, open(f"{OUT}/grip_scan.json", "w"))
    for k, v in res["boundaries"].items():
        n = sum(v.values())
        print(k, n, {kk: f"{vv} ({vv / n:.1%})" for kk, vv in sorted(v.items())})


if __name__ == "__main__":
    main()
