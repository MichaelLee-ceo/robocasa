"""Decision points (rows in use) whose settled gripper ramp is probably on the wrong side, from the main-run states and
traces (no sim; the main run's commands are not saved, so these are flags, not certainties):
  t0          t = 0: true ramp 0, settle sets +1 (every rollout first closes the fingers 41 -> 24 mm)
  creep_v4    v4 rows, gap 75.2-78.5 mm, widening over the last 5 steps, nothing grasped (slow end of an opening)
  wide_v3     v3-or-older rows, gap >= 75.2 mm while the trace says an object is grasped (bug-4 type)
  held_widen  an object grasped and the gap widened > 0.5 mm in the last step (read as opening by v3 and v4)
  fixture_mid fingers touch a fixture (trace fx_touch), no object touched, gap 5-75.2 mm, static (mostly an open gripper
              squeezed against the microwave door in SteamInMicrowave stage 2; v3 and v4 read it as closing)
usage: python -I points_at_risk.py -> out/points_at_risk.json"""
import gzip, json, os, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_scan import rows, R  # noqa: E402

DIRS = {"KettleBoiling": ["run_v2/s0", "run_v2/s1", "run_v2/s2", "run_v2/s3", "run_v2/s4", "run_dense_KettleBoiling", "run_confirm_KettleBoiling"],
        "SteamInMicrowave": ["run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix3", "run_dense_SteamInMicrowave", "run_dense2_SteamInMicrowave"],
        "PackIdenticalLunches": ["run_PackIdenticalLunches", "run_PackIdenticalLunches_fix3", "run_dense_PackIdenticalLunches", "run_dense2_PackIdenticalLunches"]}
OPEN_V3, OPEN_V4 = abs(0.020833 * 2) * 1.9 - 0.004, 0.0785


def main():
    out = defaultdict(list)
    cnt = Counter()
    for task, dirs in DIRS.items():
        cache = {}
        for d in dirs:
            for r in rows(d):
                tag, t = r["tag"], r["t"]
                if tag not in cache:
                    cache[tag] = (json.load(gzip.open(f"{R}/20261025_seen480/analysis/{task}/{tag}_trace.json.gz")),
                                  np.load(f"{R}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"])
                tr, ep = cache[tag]
                g = np.abs(ep[:, 12] - ep[:, 13])
                gap = g[t]
                v1 = gap - g[t - 1] if t else 0.0
                d5 = gap - g[t - 5] if t >= 5 else 0.0
                grasp = any(tr["obj"][o]["grasp"][t] for o in tr["obj"])
                touch = any(tr["obj"][o]["touch"][t] for o in tr["obj"])
                fx = any(v[t] for v in tr["fx_touch"].values())
                ver = r.get("hidden")
                dn = sum(ro["done"] for c in r["cands"] for ro in c["rollouts"]); n = sum(len(c["rollouts"]) for c in r["cands"])
                rec = dict(dir=d, tag=tag, t=t, k=r["k"], hidden=ver, gap_mm=round(gap * 1000, 1), V=f"{dn}/{n}")
                cats = []
                if t == 0:
                    cats.append("t0")
                if ver == "settled-v4" and OPEN_V3 <= gap < OPEN_V4 and d5 > 0.0001 and not grasp:
                    cats.append("creep_v4")
                if ver != "settled-v4" and gap >= OPEN_V3 and grasp:
                    cats.append("wide_v3")
                if grasp and v1 > 0.0005:
                    cats.append("held_widen")
                if fx and not touch and 0.005 <= gap < OPEN_V3 and abs(d5) < 0.0005:
                    cats.append("fixture_mid")
                for c in cats:
                    out[c].append(rec); cnt[(task, d, c)] += 1
    for k, v in sorted(cnt.items()):
        print(k, v)
    for c, lst in out.items():
        if c != "t0":
            print(f"\n{c}:")
            for x in lst[:40]:
                print("  ", x)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "points_at_risk.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
