"""Shared definitions for the data-integrity review (read-only on the data)."""
import glob, json, os

R = "/home/michaellee/mclee/robocasa"
DATA = f"{R}/runs/20261026_critic_data"
SEEN = f"{R}/runs/20261025_seen480"
HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]
N_STAGES = {"KettleBoiling": 2, "SteamInMicrowave": 4, "PackIdenticalLunches": 4}
L_TABLE = {"KettleBoiling": [416, 272], "SteamInMicrowave": [432, 720, 576, 416], "PackIdenticalLunches": [416, 576, 608, 496]}

# coarse override chain (effect_by_event.points()): later levels replace earlier ones for the same (tag, t)
COARSE = {
    "KettleBoiling": [["run_v2/s0", "run_v2/s1", "run_v2/s2", "run_v2/s3", "run_v2/s4"], ["run_KettleBoiling_fix"], []],
    "SteamInMicrowave": [["run_SteamInMicrowave_v2"], ["run_SteamInMicrowave_fix"], ["run_SteamInMicrowave_fix3"]],
    "PackIdenticalLunches": [["run_PackIdenticalLunches"], ["run_PackIdenticalLunches_fix"], ["run_PackIdenticalLunches_fix3"]],
}
COARSE_JOBS = {"KettleBoiling": "jobs.json", "SteamInMicrowave": "jobs_SteamInMicrowave.json",
               "PackIdenticalLunches": "jobs_PackIdenticalLunches.json"}
FIX_JOBS = {  # jobs file -> dir whose row must be current for those points
    "jobs_fix_KettleBoiling.json": "run_KettleBoiling_fix",
    "jobs_fix_SteamInMicrowave.json": "run_SteamInMicrowave_fix",
    "jobs_fix_PackIdenticalLunches.json": "run_PackIdenticalLunches_fix",
    "jobs_fix3_SteamInMicrowave.json": "run_SteamInMicrowave_fix3",
    "jobs_fix3_PackIdenticalLunches.json": "run_PackIdenticalLunches_fix3",
    "jobs_fix4_SteamInMicrowave.json": "run_SteamInMicrowave_fix3",
}
# other dirs: role, task, jobs file, expected hidden tag (code version at collection time)
OTHER = {
    "run_dense_KettleBoiling": ("dense", "KettleBoiling", "jobs_dense_KettleBoiling.json", "settled-basefix"),
    "run_dense_SteamInMicrowave": ("dense", "SteamInMicrowave", "jobs_dense_SteamInMicrowave.json", "settled-v3"),
    "run_dense_PackIdenticalLunches": ("dense", "PackIdenticalLunches", "jobs_dense_PackIdenticalLunches.json", "settled-v3"),
    "run_dense_SteamInMicrowave_bug3": ("bug3", "SteamInMicrowave", None, "settled-basefix"),
    "run_dense_PackIdenticalLunches_bug3": ("bug3", "PackIdenticalLunches", None, "settled-basefix"),
    "run_confirm_KettleBoiling": ("confirm", "KettleBoiling", "jobs_confirm_KettleBoiling.json", "settled-v3"),
    "run_dense2_SteamInMicrowave": ("dense2", "SteamInMicrowave", "jobs_dense2_SteamInMicrowave.json", "settled-v4"),
    "run_dense2_PackIdenticalLunches": ("dense2", "PackIdenticalLunches", "jobs_dense2_PackIdenticalLunches.json", "settled-v4"),
    # bug-1 positive controls (collected before the 2026-10-27 fix; superseded long ago)
    "run": ("bug1", "KettleBoiling", "jobs.json", None),
    "run_SteamInMicrowave": ("bug1", "SteamInMicrowave", "jobs_SteamInMicrowave.json", None),
}
COARSE_HIDDEN = {0: "settled", 1: "settled-basefix", 2: "settled-v3"}


def coarse_dirs():
    out = {}
    for task, levels in COARSE.items():
        for lvl, ds in enumerate(levels):
            for d in ds:
                out[d] = (task, lvl)
    return out


def read_rows(d):
    """every line of results_w*.jsonl in DATA/d: (file, line number, row or None if it does not parse)"""
    out = []
    for p in sorted(glob.glob(f"{DATA}/{d}/results_w*.jsonl")):
        for i, line in enumerate(open(p)):
            try:
                out.append((os.path.basename(p), i, json.loads(line)))
            except json.JSONDecodeError:
                out.append((os.path.basename(p), i, None))
    return out


def scenes():
    sc = {}
    for f in glob.glob(f"{DATA}/jobs*.json"):
        for j in json.load(open(f)):
            sc[j["tag"]] = j["scene"]
    return sc
