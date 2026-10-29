"""Decision points for collect.py: the 30 seed-0 episodes of a task in runs/20261025_seen480, every EVERY steps
(a chunk boundary) before the episode ended; stage k at t from the episode's stage history.
N = 8 candidates, M = 2 continuations, L per stage as in the search (runs/20261026_imagine).
Writes jobs/<task>.json (KettleBoiling: EVERY 80; (EVERY 160 by default: their episodes are
1.3-2.5x longer). Also a smoke job file (one decision point, N = 2, short L), from the deepest stage reached.
usage: python make_jobs.py [task=KettleBoiling] [every]"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "20261025_seen480", "run")
L = {"KettleBoiling": [416, 272], "SteamInMicrowave": [432, 720, 576, 416], "PackIdenticalLunches": [416, 576, 608, 496]}
TASK = sys.argv[1] if len(sys.argv) > 1 else "KettleBoiling"
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else (80 if TASK == "KettleBoiling" else 160)

scene = {j["tag"]: j["scene"] for j in json.load(open(f"{RUN}/jobs.json"))}
eps = [json.loads(l) for p in glob.glob(f"{RUN}/results_w*.jsonl") for l in open(p)]
eps = sorted([r for r in eps if r.get("task") == TASK and "error" not in r],
             key=lambda r: int(r["tag"].split("_s")[1].split("_")[0]))
jobs = []
for r in eps:
    for t in range(0, r["steps"], EVERY):
        k = sum(tk <= t for tk, _ in r["history"])
        if k >= len(L[TASK]):
            continue
        jobs.append(dict(task=TASK, tag=r["tag"], scene=scene[r["tag"]], t=t, k=k, L=L[TASK][k], N=8, M=2,
                         episode_success=r["success"]))
name = "" if TASK == "KettleBoiling" else f"_{TASK}"
json.dump(jobs, open(f"{HERE}/jobs/{TASK}.json", "w"), indent=0)
smoke = jobs[1] if TASK == "KettleBoiling" else max(jobs, key=lambda j: (j["k"], -j["t"]))
json.dump([dict(smoke, N=2, L=48)], open(f"{HERE}/jobs/smoke_{TASK}.json", "w"), indent=0)
print(TASK, len(jobs), "decision points; per stage", [sum(j["k"] == k for j in jobs) for k in range(len(L[TASK]))],
      ";", sum(not j["episode_success"] for j in jobs), "from failed episodes")
