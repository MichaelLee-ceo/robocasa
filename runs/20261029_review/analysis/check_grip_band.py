"""Every collected point whose main-run finger gap at t lies in the band where the old and the new settle() rules disagree
(75.2 mm <= gap < 78.5 mm, not widening by > 0.5 mm in the last step): old -> "opening", new (settled-v4) -> "closing".
The main run's gap over the 4 steps before t shows which side the gripper was really commanded to (the ramp at t is the
result of those commands): falling by > 0.5 mm -> closing, rising -> opening, flat -> blocked (e.g. a wide grasp, closing).
Lists each such row with the rule it was collected with. Read-only. usage: python -I check_grip_band.py"""
import collections, glob, gzip, json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
SEEN = f"{R}/20261025_seen480"
sys.path.insert(0, f"{R}/20261028_critical")
from effect_by_event import RUNS, points  # noqa: E402

OLD, NEW = 0.0752, 0.0785


def main():
    tr = {}
    rows = [("coarse", p["task"], p["tag"], p["t"], p["hidden"]) for p in points()]
    for task in RUNS:
        for d in (f"run_dense_{task}", f"run_dense2_{task}", f"run_confirm_{task}"):
            for f in glob.glob(f"{DATA}/{d}/results_w*.jsonl"):
                for line in open(f):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" not in r:
                        rows.append((d.split("_")[1] if "dense" in d or "confirm" in d else d, task, r["tag"], r["t"], r.get("hidden")))
    # pending confirmation jobs (collected with v4 once they run)
    for task in ["SteamInMicrowave", "PackIdenticalLunches"]:
        f = f"{DATA}/jobs_confirm_{task}.json"
        if os.path.exists(f):
            rows += [("confirm-job", task, j["tag"], j["t"], "settled-v4") for j in json.load(open(f))]
    out = collections.Counter()
    for kind, task, tag, t, hidden in rows:
        if tag not in tr:
            tr[tag] = json.load(gzip.open(f"{SEEN}/analysis/{task}/{tag}_trace.json.gz"))["grip_width"]
        g = tr[tag]
        if not t or not (OLD <= g[t] < NEW) or g[t] - g[t - 1] > 0.0005:
            continue
        trend = g[t] - g[max(0, t - 4)]
        truth = "closing" if trend < -0.0005 else ("opening" if trend > 0.0002 else "flat")
        used = "closing" if hidden == "settled-v4" else "opening"
        wrong = (truth == "closing" and used == "opening") or (truth == "opening" and used == "closing")
        out[(kind, truth, used, wrong)] += 1
        print(f"{kind:12s} {task[:5]} {tag.split('_')[1]:4s} t{t:5d} hidden {hidden!s:16s} gap t-4..t+4 (mm) "
              f"{[round(1000 * x, 1) for x in g[max(0, t - 4):t + 5]]} -> main run {truth:7s}, ramp set {used:7s}{'   WRONG' if wrong else ''}")
    print("summary (kind, main run before t, ramp set, wrong):")
    for k, v in sorted(out.items()):
        print("  ", k, v)


if __name__ == "__main__":
    main()
