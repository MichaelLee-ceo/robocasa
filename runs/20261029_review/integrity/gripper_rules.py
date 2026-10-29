"""How settle()'s gripper-ramp inference compares with what the main run's fingers were doing, at every scanned point.

The finger gap of the main run around t (t-4 .. t+2) shows which way the gripper was commanded at t: a gripper told to
open keeps opening towards the limit (slowly near it), one told to close closes or stays blocked on an object.
For each point: the old rule (pre-v4: opening if gap >= 1.9 x init gap - 4 mm = 75.2 mm, or widening > 0.5 mm in
the last step), the v4 rule (opening if gap >= 80 - 1.5 = 78.5 mm, or widening > 0.5 mm in the last step), the rule
that was actually used (from the row's hidden tag), and per rollout the first-chunk signature:
  opened_on_close: the candidate commands close for its first 5 steps, but the gap grows by > 1 mm (ramp started open)
  closed_on_open:  the candidate commands open for its first 5 steps, but the gap shrinks by > 5 mm (ramp started
                   closed; 5 mm excludes the small contact-driven closings seen in the main runs)
Writes gripper_rules.json and prints tables. usage: python -I gripper_rules.py"""
import collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEEN, HERE  # noqa: E402
from analyze import load  # noqa: E402

OLD_OPEN = abs(0.020833 - -0.020833) * 1.9
NEW_OPEN = 0.08


def rule(gt, gtm1, version):
    widening = (gt - gtm1 > 0.0005) if gtm1 is not None else False
    thr = OLD_OPEN - 0.004 if version == "old" else NEW_OPEN - 0.0015
    return "closing" if (gt < thr and not widening) else "opening"


def main():
    recs, _ = load()
    eps = {}
    out, tab = [], collections.defaultdict(collections.Counter)
    for r in recs:
        key = (r["task"], r["tag"])
        if key not in eps:
            s = np.load(f"{SEEN}/run/{r['task']}/{r['tag']}_steps.npz")["states"]
            eps[key] = np.abs(s[:, 12] - s[:, 13])  # finger joints at qpos 11, 12 in every scene (layout.json)
        g = eps[key]; t = r["t"]
        gt = g[t]; gtm1 = g[t - 1] if t else None
        d4 = gt - g[max(t - 4, 0)]
        nxt = g[min(t + 2, len(g) - 1)] - gt
        # the main run's commanded side at t, from its fingers: still opening (rising before t, or rising right after t
        # by > 0.2 mm), or closing (falling right after t by > 0.2 mm); otherwise steady (blocked or at a limit)
        if d4 > 0.0002 or nxt > 0.0002:
            main_side = "opening"
        elif nxt < -0.0002:
            main_side = "closing"
        else:
            main_side = "steady"
        used = "v4" if r["hidden"] == "settled-v4" else ("none" if r["hidden"] is None else "old")
        inferred = rule(gt, gtm1, "new" if used == "v4" else "old") if used != "none" else None
        ro = [o for o in r["rolls"] if "gap" in o]
        oc = sum(all(x >= 0.5 for x in o["gcmd0"][:5]) and (max(o["gap"][1:6]) - o["gap"][0]) > 0.001 for o in ro)
        co = sum(all(x < 0.5 for x in o["gcmd0"][:5]) and (o["gap"][0] - min(o["gap"][1:6])) > 0.005 for o in ro)
        nc = sum(all(x >= 0.5 for x in o["gcmd0"][:5]) for o in ro)
        no = sum(all(x < 0.5 for x in o["gcmd0"][:5]) for o in ro)
        rec = dict(group=f"{r['status']}:{r['dir']}", tag=r["tag"], t=t, k=r["k"], hidden=r["hidden"], used=used,
                   gap_mm=round(gt * 1000, 2), d1_mm=round((gt - gtm1) * 1000, 2) if gtm1 is not None else None,
                   d4_mm=round(d4 * 1000, 2), next2_mm=round(nxt * 1000, 2), main_side=main_side, inferred=inferred,
                   old_rule=rule(gt, gtm1, "old"), v4_rule=rule(gt, gtm1, "new"),
                   n_close_cmd=nc, opened_on_close=oc, n_open_cmd=no, closed_on_open=co,
                   done=sum(o["done"] for o in r["rolls"]), n=len(r["rolls"]))
        out.append(rec)
        grp = rec["group"]
        tab[grp]["points"] += 1
        if inferred and main_side != "steady" and inferred != main_side:
            tab[grp][f"inferred_{inferred}_main_{main_side}"] += 1
        if main_side != "steady" and rec["v4_rule"] != main_side:
            tab[grp][f"v4_would_say_{rec['v4_rule']}_main_{main_side}"] += 1
        if main_side != "steady" and rec["old_rule"] != main_side:
            tab[grp][f"old_would_say_{rec['old_rule']}_main_{main_side}"] += 1
        if nc and oc >= max(1, nc // 2):
            tab[grp]["pts_majority_opened_on_close"] += 1
        if no and co >= max(1, no // 2):
            tab[grp]["pts_majority_closed_on_open"] += 1
    json.dump(out, open(os.path.join(HERE, "gripper_rules.json"), "w"), indent=0)
    for grp in sorted(tab):
        print(f"{grp:45s}", dict(tab[grp]))
    print("--- points whose rollouts show the ramp artifact (majority of the candidates that command one side move the other way)")
    for x in out:
        if (x["n_close_cmd"] and x["opened_on_close"] >= max(1, x["n_close_cmd"] // 2)) or (x["n_open_cmd"] and x["closed_on_open"] >= max(1, x["n_open_cmd"] // 2)):
            print(json.dumps({k: x[k] for k in ("group", "tag", "t", "k", "hidden", "gap_mm", "d1_mm", "d4_mm", "next2_mm", "main_side",
                                                 "inferred", "n_close_cmd", "opened_on_close", "n_open_cmd", "closed_on_open", "done", "n")}))


if __name__ == "__main__":
    main()
