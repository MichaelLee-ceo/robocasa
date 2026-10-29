"""Decision points to re-collect with the settled-v5 restore, from affected_<task>.jsonl (affected.py), for the rows in use
(override order coarse: base -> fix -> fix3; dense, dense2 and confirm stand alone):
  - every t = 0 point (v5 starts from the exact hidden state; all earlier rows set the gripper ramp to +-1, truth 0);
  - every point whose restored gripper ramp changes (diff_steps >= 1);
  - rows from before the bug-3 fix ('settled', 'settled-basefix') whose base had turned >= 2 degrees since the episode
    start (their base commands were rotated by that angle; fix3 re-collected >= 5 degrees only).
Confirm points are left for phase 2 (after the coarse and dense data are re-analysed). Writes points_v5.json (all
selected rows) and the jobs file. usage: python select_v5.py jobs_v5.json"""
import json, sys
from collections import Counter, defaultdict

D = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"
DYAW = json.load(open("/home/michaellee/mclee/robocasa/runs/20261028_critical/dyaw.json"))
ORDER = {"KettleBoiling": [("coarse", ["run_v2/s0", "run_v2/s1", "run_v2/s2", "run_v2/s3", "run_v2/s4", "run_KettleBoiling_fix"]),
                           ("dense", ["run_dense_KettleBoiling"]), ("confirm", ["run_confirm_KettleBoiling"])],
         "SteamInMicrowave": [("coarse", ["run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix", "run_SteamInMicrowave_fix3"]),
                              ("dense", ["run_dense_SteamInMicrowave"]), ("dense2", ["run_dense2_SteamInMicrowave"]),
                              ("confirm", ["run_confirm_SteamInMicrowave"])],
         "PackIdenticalLunches": [("coarse", ["run_PackIdenticalLunches", "run_PackIdenticalLunches_fix", "run_PackIdenticalLunches_fix3"]),
                                  ("dense", ["run_dense_PackIdenticalLunches"]), ("dense2", ["run_dense2_PackIdenticalLunches"]),
                                  ("confirm", ["run_confirm_PackIdenticalLunches"])]}
OUT = {"coarse": "run_{task}_fix5", "dense": "run_dense_{task}_fix5", "dense2": "run_dense2_{task}_fix5", "confirm": "run_confirm_{task}_fix5"}


def main():
    jobs_out = sys.argv[1]
    sel, stats = [], Counter()
    for task, groups in ORDER.items():
        rows = [json.loads(l) for l in open(f"out/affected_{task}.jsonl")]
        by_dir = defaultdict(dict)
        for r in rows:
            by_dir[r["dir"]][(r["tag"], r["t"])] = r
        for ds, dirs in groups:
            use = {}
            for d in dirs:  # later dirs replace earlier ones
                use.update(by_dir.get(d, {}))
            for (tag, t), r in sorted(use.items()):
                dyaw = DYAW.get(task, {}).get(f"{tag}|{t}")
                why = []
                if t == 0:
                    why.append("t0")
                if r["diff_steps"] >= 1:
                    why.append(f"ramp{r['diff_steps']}")
                if r["hidden"] in ("settled", "settled-basefix") and dyaw is not None and abs(dyaw) >= 2:
                    why.append("yaw")
                stats[(task, ds, "n")] += 1
                if not why:
                    continue
                stats[(task, ds, "redo")] += 1
                for w in why:
                    stats[(task, ds, "ramp" if w.startswith("ramp") else w)] += 1
                sel.append(dict(task=task, dataset=ds, dir=r["dir"], tag=tag, t=t, why=why, dyaw=dyaw, old=r["old"], new=r["new"],
                                N=r["N"], M=r["M"], L=r["L"], wall_s=r["wall_s"], base_moved=r["base_moved"],
                                out=f"{D}/" + OUT[ds].format(task=task)))
    for task in ORDER:
        for ds in ("coarse", "dense", "dense2", "confirm"):
            if stats[(task, ds, "n")]:
                print(f"{task:22s} {ds:8s} in use {stats[(task, ds, 'n')]:4d}  redo {stats[(task, ds, 'redo')]:4d}  "
                      f"(t0 {stats[(task, ds, 't0')]}, ramp {stats[(task, ds, 'ramp')]}, yaw {stats[(task, ds, 'yaw')]})")
    json.dump(sel, open("points_v5.json", "w"), indent=1)
    ph1 = [p for p in sel if p["dataset"] != "confirm"]
    print("phase 1 points:", len(ph1), " old wall-s sum:", round(sum(p["wall_s"] or 0 for p in ph1)),
          " confirm (phase 2):", len(sel) - len(ph1))
    sc = {}
    for p_ in __import__("glob").glob(f"{D}/jobs*.json"):
        for j in json.load(open(p_)):
            sc.setdefault(j["tag"], j["scene"])
    jobs = []
    for p in ph1:
        r = None
        for l in (l for f in __import__("glob").glob(f"{D}/{p['dir']}/results_w*.jsonl") for l in open(f)):
            try:
                x = json.loads(l)
            except json.JSONDecodeError:
                continue
            if x.get("tag") == p["tag"] and x.get("t") == p["t"] and "error" not in x:
                r = x
        j = {k: r[k] for k in ("task", "tag", "t", "k", "L", "N", "M", "episode_success")}
        j.update(scene=sc[p["tag"]], out=p["out"], salt="", replaces=p["dir"], est_s=p["wall_s"] or 600)
        jobs.append(j)
    # claimed from the end: task blocks (few env rebuilds per worker), the longest jobs last so they start first
    tord = {"KettleBoiling": 0, "SteamInMicrowave": 1, "PackIdenticalLunches": 2}
    jobs.sort(key=lambda j: (tord[j["task"]], j["est_s"]))
    json.dump(jobs, open(jobs_out, "w"), indent=1)
    print(len(jobs), "jobs ->", jobs_out)


if __name__ == "__main__":
    main()
