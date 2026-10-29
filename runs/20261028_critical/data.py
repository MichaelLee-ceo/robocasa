"""The collected decision points (../20261026_critic_data/data), shared by every analysis here and by
../20261026_critic_data/analyze_phase.py.

Layout: data/<task>/<dataset>/results.jsonl (one row per decision point; later collections add results_w<W>.jsonl), roll/ (rollouts: <tag>_t<t>_c<i>_m<m>.npz and
.mp4) and snap/ (<tag>_t<t>.npz: cameras and the exact policy request at the point). Datasets:
  coarse   uniform decision points along the main runs, 8 candidates x 2 continuations
  dense    V points inside the intervals where the coarse V jumps (and their refinement to single chunks), 8 x 1
  confirm  the located critical chunks, 8 x 4, salted continuation seeds
  control  random uniform decision points, same protocol as confirm (the control for it)
Every row was collected with the settled-v5 restore or with an earlier restore that gives the same hidden state at that
point. row["hidden"] and row["src_dir"] say which, and ../20261029_review/fix/README.md explains the history (the
2026-10-29 reorganization dropped every superseded row). row["_dir"] is the dataset dir. A row's rollouts (roll()) and
snapshot (snap()) are in the same dir.
The analyses write to $CRIT_OUT (default: this folder).
usage: python -I data.py   (prints the points per task and dataset, with their hidden tags)"""
import collections, functools, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "20261026_critic_data"))
OUT = os.environ.get("CRIT_OUT", HERE)
TASKS = ("KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches")
DATASETS = ("coarse", "dense", "confirm", "control")
JOBS = {t: f"jobs/{t}.json" for t in TASKS}  # the coarse jobs: scene, stage k and budget L of every point (job templates)


def warn(msg):
    print(f"data.py: {msg}", file=sys.stderr)


def rel(d):
    return os.path.relpath(d, DATA)


def ddir(task, ds="coarse"):
    assert task in TASKS and ds in DATASETS, (task, ds)
    return os.path.join(DATA, "data", task, ds)


@functools.lru_cache(maxsize=None)
def _read(task, ds):
    d, use = ddir(task, ds), {}
    # results.jsonl (the 2026-10-29 reorganization), then results_w<W>.jsonl of any later collection into the same dir
    for p in sorted(glob.glob(os.path.join(d, "results*.jsonl")), key=lambda p: (not p.endswith("results.jsonl"), p)):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line a worker is still writing
                continue
            if "error" in r:
                continue
            r["_dir"] = d
            key = (r["tag"], r["t"])
            if key in use:
                warn(f"{rel(d)}: two rows of {key}")
            use[key] = r
    return use


def rows(task, ds="coarse"):
    """{(tag, t): row}; row["_dir"] is the dataset dir"""
    return dict(_read(task, ds))


def all_rows(task, ds="coarse"):
    """every row (kept for the audits that used to see superseded layers too; there are none left)"""
    return [dict(r, _in_use=True) for r in _read(task, ds).values()]


def roll(r):
    """the row's rollouts: roll(r)/<tag>_t<t>_c<i>_m<m>.npz"""
    return os.path.join(r["_dir"], "roll")


def snap(r):
    return os.path.join(r["_dir"], "snap", f"{r['tag']}_t{r['t']}.npz")


def outcomes(r):
    """candidates x continuations array of done (stage k finished)"""
    return np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float)


@functools.lru_cache(maxsize=None)
def scenes():
    """{tag: scene file}; the scene is the episode's, the same at every t"""
    return {j["tag"]: j["scene"] for f in JOBS.values() for j in json.load(open(os.path.join(DATA, f)))}


def main():
    for task in TASKS:
        for ds in DATASETS:
            use = rows(task, ds)
            print(f"{task:21s} {ds:8s} {len(use):4d} points  hidden {dict(collections.Counter(r.get('hidden') for r in use.values()))}")


if __name__ == "__main__":
    main()
