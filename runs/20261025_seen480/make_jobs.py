"""Job list for the Composite-Seen rerun: 16 tasks x 30 frozen scenes, original horizons, official composite_seen
checkpoint with the official task instruction (baseline_seed), seeded action noise (policy_seed 0) and every control
step's sim state saved (save_steps). usage: python make_jobs.py <out jobs.json> [comma-separated tasks] [seeds e.g. 0-29]"""
import json, os, sys

R = os.environ.get("ROBOCASA_ROOT", "/home/michaellee/mclee/robocasa") + "/runs/"
SEEN = ["DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher", "PackIdenticalLunches", "PreSoakPan",
        "PrepareCoffee", "RinseSinkBasin", "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation", "StackBowlsCabinet",
        "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce"]


def scene(t, s):  # s0-19 of the original 5 tasks in scenes_seen5_v2, s20-29 in _s30, the other 11 tasks in scenes_seen11_v2
    for d in ("scenes_seen5_v2", "scenes_seen5_v2_s30", "scenes_seen11_v2"):
        p = f"{R}{d}/{t}_s{s}.npz"
        if os.path.exists(p) and (d != "scenes_seen5_v2" or s < 20):
            return p
    raise SystemExit(f"no frozen scene for {t} s{s}")


out = sys.argv[1]
tasks = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else SEEN
lo, hi = map(int, sys.argv[3].split("-")) if len(sys.argv) > 3 else (0, 29)
H = json.load(open(f"{R}horizons_original_plus.json"))
jobs = [dict(task=t, seed=s, scene=scene(t, s), condition="baseline_seed", horizon=H[t], tag=f"{t}_s{s}_single",
             policy_seed=0, save_steps=True) for t in tasks for s in range(lo, hi + 1)]
json.dump(jobs, open(out, "w"), indent=1)
print(len(jobs), "jobs ->", out)
