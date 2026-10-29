"""Shared loaders for the review (read-only on the data).

Rows:
  coarse(task): the current coarse rows, with the override order of 20261028_critical/effect_by_event.py:points()
                (run_v2/s* | run_<task>_v2 | run_<task>  ->  run_<task>_fix  ->  run_<task>_fix3); every row keeps the
                list of rows it replaced (`_replaced`).
  dense(task):  run_dense_<task> (never the *_bug3 dirs) and run_dense2_<task> (complete rows only).
Main runs: history [[t_finished, k], ...] per tag; scenes per tag."""
import glob, json, os

R = "/home/michaellee/mclee/robocasa"
DATA = f"{R}/runs/20261026_critic_data"
SEEN = f"{R}/runs/20261025_seen480"
CRIT = f"{R}/runs/20261028_critical"
HERE = os.path.dirname(os.path.abspath(__file__))
TASKS3 = ["KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]
NSTAGES = {"KettleBoiling": 2, "SteamInMicrowave": 4, "PackIdenticalLunches": 4}
COARSE_DIRS = {"KettleBoiling": ["run_v2/s*", "run_KettleBoiling_fix", "run_KettleBoiling_fix3"],
               "SteamInMicrowave": ["run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix", "run_SteamInMicrowave_fix3"],
               "PackIdenticalLunches": ["run_PackIdenticalLunches", "run_PackIdenticalLunches_fix", "run_PackIdenticalLunches_fix3"]}
DENSE_DIRS = {t: [f"run_dense_{t}", f"run_dense2_{t}"] for t in TASKS3}


def read_rows(d):
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, d, "results_w*.jsonl"))):
        for i, line in enumerate(open(p)):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            r["_dir"] = os.path.relpath(os.path.dirname(p), DATA)
            r["_file"] = p
            r["_line"] = i
            out.append(r)
    return out


def coarse(task):
    pts = {}
    for d in COARSE_DIRS[task]:
        for r in read_rows(d):
            key = (r["tag"], r["t"])
            if key in pts:
                r["_replaced"] = pts[key].get("_replaced", []) + [(pts[key]["_dir"], pts[key].get("hidden"))]
            pts[key] = r
    return pts


def dense(task):
    pts = {}
    for d in DENSE_DIRS[task]:
        for r in read_rows(d):
            key = (r["tag"], r["t"])
            if key in pts:
                r["_replaced"] = pts[key].get("_replaced", []) + [(pts[key]["_dir"], pts[key].get("hidden"))]
            pts[key] = r
    return pts


def rolls(r):
    """flat list of (cand, m, rollout dict)"""
    return [(c["cand"], x["m"], x) for c in r["cands"] for x in c["rollouts"]]


def roll_path(r, c, m):
    return os.path.join(DATA, r["_dir"], "roll", f"{r['tag']}_t{r['t']}_c{c}_m{m}.npz")


def main_runs():
    rows = [json.loads(l) for p in glob.glob(f"{SEEN}/run/results_w*.jsonl") for l in open(p)]
    return {r["tag"]: r for r in rows if r.get("task") in TASKS3 and "error" not in r}


def scenes():
    out = {}
    for f in ["jobs.json", "jobs_SteamInMicrowave.json", "jobs_PackIdenticalLunches.json"]:
        for j in json.load(open(f"{DATA}/{f}")):
            out[j["tag"]] = j["scene"]
    return out


def stage_at(history, t):
    """the stage the main run is in at step t under the history semantics of make_jobs.py: stage k finished at tk is
    counted from tk on (the check at the chunk boundary tk is evaluated on the state at tk)."""
    return sum(tk <= t for tk, _ in history)
