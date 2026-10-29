"""Which row does effect_by_event.points() use for each coarse decision point, and is it the current one?
- every job of jobs_fix_<task> / jobs_fix3_<task> / jobs_fix4 must resolve to a row of the matching dir;
- every SteamInMicrowave / PackIdenticalLunches point with a base turn >= 5 deg (dyaw.json) must resolve to FIX3;
- roll_dir() (disagreement.py) must point to the same dir as the row in use;
- the glob order of run_v2/s* must not matter (no (tag, t) in two shards).
Read-only. usage: python -I check_override.py"""
import collections, glob, json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
CRIT = f"{R}/20261028_critical"
sys.path.insert(0, CRIT)
from effect_by_event import RUNS, FIX, FIX3, points  # noqa: E402
from disagreement import roll_dir  # noqa: E402


def rows(d):
    out = {}
    for p in glob.glob(os.path.join(DATA, d, "results_w*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                out.setdefault((r["tag"], r["t"]), []).append((os.path.relpath(os.path.dirname(p), DATA), r.get("hidden")))
    return out


def main():
    pts = points()
    dyaw = json.load(open(f"{CRIT}/dyaw.json"))
    src = {}
    for task in RUNS:
        for d in (RUNS[task], FIX[task], FIX3[task]):
            for key, v in rows(d).items():
                if len(v) > 1:
                    print("  duplicate", key, v)
                src[key] = v[-1]
    print(f"{len(pts)} points in use")
    by = collections.Counter((p["task"], src[(p["tag"], p["t"])][1]) for p in pts)
    for k, v in sorted(by.items()):
        print(f"  {k[0]:21s} hidden={k[1]!s:16s} {v}")
    # jobs of the fix dirs resolve to those dirs
    for task in ["KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]:
        for f, d in [(f"jobs_fix_{task}.json", FIX[task]), (f"jobs_fix3_{task}.json", FIX3[task]), (f"jobs_fix4_{task}.json", FIX3[task])]:
            if not os.path.exists(f"{DATA}/{f}"):
                continue
            js = json.load(open(f"{DATA}/{f}"))
            miss = [(j["tag"], j["t"]) for j in js if (j["tag"], j["t"]) not in rows(d)]
            later = [(j["tag"], j["t"]) for j in js if src[(j["tag"], j["t"])][0] != d]
            print(f"  {f:40s} {len(js):4d} jobs; without a row in {d}: {len(miss)}; resolved to another dir: {len(later)}")
    # every point with a base turn >= 5 degrees resolves to FIX3
    for task in ["SteamInMicrowave", "PackIdenticalLunches"]:
        need = [k for k, v in dyaw[task].items() if v >= 5]
        bad = [k for k in need if src[(k.split("|")[0], int(k.split("|")[1]))][0] != FIX3[task]]
        print(f"  {task}: {len(need)} points turned >= 5 deg; not served from {FIX3[task]}: {len(bad)} {bad[:5]}")
    # roll_dir agrees with the row in use
    mism = []
    for p in pts:
        d = os.path.relpath(roll_dir(p["task"], p["tag"], p["t"]), DATA)
        if os.path.dirname(d) != src[(p["tag"], p["t"])][0]:
            mism.append((p["tag"], p["t"], d, src[(p["tag"], p["t"])][0]))
    print(f"  roll_dir vs the row in use: {len(mism)} mismatches {mism[:5]}")
    # roll dir in points() (p['roll']) vs roll_dir()
    mism2 = [(p["tag"], p["t"]) for p in pts if os.path.normpath(p["roll"]) != os.path.normpath(roll_dir(p["task"], p["tag"], p["t"]))]
    print(f"  points()['roll'] vs roll_dir(): {len(mism2)} mismatches {mism2[:5]}")


if __name__ == "__main__":
    main()
