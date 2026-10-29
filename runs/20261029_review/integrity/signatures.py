"""Physical signatures of bugs 2 and 3 in the saved rollouts (no simulation), per group (status:dir).

Bug 2 (arm lunge in base mode): the largest arm joint speed over the first 16 steps, for rollouts whose first chunk
starts in base mode (control_mode >= 0.5) vs arm mode.
Bug 3 (base commands turned by the yaw since the start): per rollout, the signed median angle off_c from the commanded
base velocity, turned as the controller does with the episode's start pose, to the realized slide-joint velocity
(scan.py base_dir_test). Under bug 3 it is about -dyaw (the base yaw turned between the episode start and t), else
about 0. Per point the median over rollouts with >= 10 usable steps; classified with angles wrapped to (-180, 180]:
bug3-like if |wrap(off + dyaw)| < |wrap(off)| - 5, correct-like if the reverse, else ambiguous.
Writes signatures.json. usage: python -I signatures.py"""
import collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [(5, 20), (20, 45), (45, 90), (90, 400)]
w = lambda a: (a + 180) % 360 - 180


def main():
    recs, _ = load()
    b2 = collections.defaultdict(lambda: {"base": [], "arm": []})
    b3 = collections.defaultdict(collections.Counter)
    b3_list = collections.defaultdict(list)
    for r in recs:
        g = f"{r['status']}:{r['dir']}"
        for o in r["rolls"]:
            if o.get("armv16") is not None:
                b2[g]["base" if o["cm0"] >= 0.5 else "arm"].append(o["armv16"])
        dy = r["main"]["dyaw_deg"]
        offs = [o["base"]["off_c"] for o in r["rolls"] if o.get("base") and o["base"].get("n", 0) >= 10 and "off_c" in o["base"]]
        if len(offs) < 3 or abs(dy) < 5:
            continue
        off = float(np.median(offs))
        dc, db = abs(w(off)), abs(w(off + dy))
        cls = "bug3" if db < dc - 5 else ("ok" if dc < db - 5 else "amb")
        b = [f"{a}-{c}" for a, c in BINS if a <= abs(dy) < c][0]
        b3[g][f"{b}:{cls}"] += 1
        b3_list[g].append(dict(tag=r["tag"], t=r["t"], dyaw=round(dy, 1), off_c=round(off, 1), n=len(offs), cls=cls, hidden=r["hidden"]))
    out = dict(bug2={g: {m: dict(n=len(v), p50=float(np.median(v)) if v else None, p95=float(np.percentile(v, 95)) if v else None,
                                 max=float(max(v)) if v else None, frac_gt_3=float(np.mean(np.array(v) > 3)) if v else None)
                         for m, v in d.items()} for g, d in b2.items()},
               bug3={g: dict(v) for g, v in b3.items()}, bug3_points=b3_list)
    json.dump(out, open(os.path.join(HERE, "signatures.json"), "w"), indent=1)
    print("bug 2: max arm joint speed (rad/s) over the first 16 steps")
    for g in sorted(out["bug2"]):
        for m, s in out["bug2"][g].items():
            if s["n"]:
                print(f"  {g:45s} {m:4s} n {s['n']:5d} p50 {s['p50']:.2f} p95 {s['p95']:.2f} max {s['max']:.2f} >3 rad/s {s['frac_gt_3']:.3f}")
    print("bug 3: points with |dyaw| >= 5 deg by |dyaw| bin: ok / bug3 / ambiguous")
    for g in sorted(out["bug3"]):
        c = out["bug3"][g]
        print(f"  {g:45s}", "  ".join(f"{a}-{b}: {c.get(f'{a}-{b}:ok', 0)}/{c.get(f'{a}-{b}:bug3', 0)}/{c.get(f'{a}-{b}:amb', 0)}" for a, b in BINS))


if __name__ == "__main__":
    main()
