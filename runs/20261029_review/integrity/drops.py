"""Systematic early drops: at a point, the share of rollouts in which a free object falls >= 10 cm within the first 16
steps while the main run's same object does not fall >= 5 cm over t .. t+16. One rollout dropping is policy behaviour;
most rollouts of a point dropping what the main run holds is the signature bugs 2 and 4 left.
Writes drops.json. usage: python -I drops.py"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
recs, _ = load()
tab = collections.defaultdict(collections.Counter)
flag = collections.defaultdict(list)
for r in recs:
    g = f"{r['status']}:{r['dir']}"
    tab[g]["points"] += 1
    best = None
    for ob, mz in r["main"]["objz"].items():
        if mz[0] - min(mz[1:]) >= 0.05:
            continue
        n = sum(1 for o in r["rolls"] if "objz" in o and (o["objz"][ob][0] - min(o["objz"][ob][1:4])) >= 0.10)
        if best is None or n > best[1]:
            best = (ob, n)
    if best and best[1]:
        tab[g]["points_with_any_drop"] += 1
        tab[g]["rollouts_dropping"] += best[1]
        if best[1] >= len(r["rolls"]) / 2:
            tab[g]["points_majority_drop"] += 1
            flag[g].append(dict(tag=r["tag"], t=r["t"], k=r["k"], obj=best[0], dropped=best[1], n=len(r["rolls"]), hidden=r["hidden"],
                                done=sum(o["done"] for o in r["rolls"])))
json.dump(dict(table={g: dict(v) for g, v in tab.items()}, flagged=flag), open(os.path.join(HERE, "drops.json"), "w"), indent=1)
for g in sorted(tab):
    print(f"{g:45s}", dict(tab[g]))
for g in sorted(flag):
    for x in flag[g]:
        print(g, json.dumps(x))
