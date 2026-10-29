"""Counts, from gripper_rules.json, the points whose rollouts start with a gripper-ramp artifact: most of the
candidates that command one side for their first 5 steps move the fingers the other way, while the main run's own
fingers right after t (t .. t+2) do not move that way (so the true ramp did not point there).
  A (closed while commanded open): >= half of the open-commanding candidates close > 5 mm; main run next2 > -2 mm
  B (opened while commanded close): >= half of the close-commanding candidates open > 1 mm; main run next2 < +0.5 mm
    and not widening before t (d1 <= 0.5 mm)
usage: python -I gripper_artifacts.py"""
import collections, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "gripper_rules.json")))
tab = collections.defaultdict(collections.Counter)
lists = collections.defaultdict(list)
for x in G:
    a = x["n_open_cmd"] >= 2 and x["closed_on_open"] >= x["n_open_cmd"] / 2 and x["next2_mm"] > -2
    b = x["n_close_cmd"] >= 2 and x["opened_on_close"] >= x["n_close_cmd"] / 2 and x["next2_mm"] < 0.5 and (x["d1_mm"] or 0) <= 0.5
    g = x["group"]
    tab[g]["points"] += 1
    tab[g]["t0_points"] += x["t"] == 0
    if a:
        tab[g]["A_closed_on_open" + ("_t0" if x["t"] == 0 else "")] += 1
    if b:
        tab[g]["B_opened_on_close" + ("_t0" if x["t"] == 0 else "")] += 1
    if (a or b) and x["t"] != 0:
        lists[g].append({k: x[k] for k in ("tag", "t", "k", "hidden", "gap_mm", "d1_mm", "d4_mm", "next2_mm", "inferred",
                                           "n_open_cmd", "closed_on_open", "n_close_cmd", "opened_on_close", "done", "n")})
out = dict(table={g: dict(v) for g, v in tab.items()}, lists=lists)
json.dump(out, open(os.path.join(HERE, "gripper_artifacts.json"), "w"), indent=1)
for g in sorted(tab):
    print(f"{g:45s}", dict(tab[g]))
for g in sorted(lists):
    print(g)
    for x in lists[g]:
        print("   ", json.dumps(x))
