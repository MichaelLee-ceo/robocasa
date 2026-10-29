"""split_points.split_points(), overlay_critical.jobs() and drop_check.jobs() take the labels from points() (RUNS -> FIX ->
FIX3) but the rollout dir (and py_state) from the RUNS dirs only. If re-run now, a re-collected point would be replayed
from its superseded rollouts while being labelled by the new ones. This counts the affected points and shows that the
old and new rollouts of such a point really differ (winners / losers by candidate). Read-only.
usage: python -I check_latent_rolls.py"""
import glob, json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
sys.path.insert(0, f"{R}/20261028_critical")
import numpy as np  # noqa: E402
from effect_by_event import RUNS, points  # noqa: E402
from disagreement import roll_dir  # noqa: E402


def main():
    old = {}
    for task, d in RUNS.items():
        for f in glob.glob(f"{DATA}/{d}/results_w*.jsonl"):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in r:
                    old[(r["tag"], r["t"])] = (np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float), os.path.join(os.path.dirname(f), "roll"))
    pts = points()
    split = [p for p in pts if (p["y"].sum(1) == 2).any() and (p["y"].sum(1) == 0).any()]
    strict = [p for p in split if (p["y"].sum(1) == 2).sum() >= 2 and (p["y"].sum(1) == 0).sum() >= 2]
    for name, g in [("split points (split_points.py)", split), ("strict points (overlay_critical.py candidates)", strict)]:
        bad = [p for p in g if os.path.normpath(old[(p["tag"], p["t"])][1]) != os.path.normpath(roll_dir(p["task"], p["tag"], p["t"]))]
        flips = 0
        for p in bad:
            yo = old[(p["tag"], p["t"])][0].sum(1); yn = p["y"].sum(1)
            flips += int(((yo == 2) & (yn == 0)).sum() + ((yo == 0) & (yn == 2)).sum())
        print(f"{name}: {len(g)} now; {len(bad)} would be replayed from superseded rollouts "
              f"({', '.join(t + ' ' + str(sum(q['task'] == t for q in bad)) for t in RUNS)}); "
              f"winner <-> loser swaps between the old and new rollouts of those points: {flips} candidates")


if __name__ == "__main__":
    main()
