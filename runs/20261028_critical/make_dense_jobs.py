"""Dense V points, to locate the chunk where a stage's outcome gets decided (the critical chunk).

V(t) = P(finish the current stage within its budget L | the main run's state at t, plain policy). At the collected
points (the coarse rows in use, data.py) it is the completion rate of their 16 rollouts. The outcome is decided where V
jumps. This script lists the intervals of the main runs that contain a jump:
  (a) two consecutive points of the same stage with |V(t1) - V(t0)| >= JUMP;
  (b) the main run finished the stage at tc, but V at its last point before tc was <= RISE_FROM. The rise to 1 happened
      in between.
Inside each interval it places new points at the main run's chunk boundaries: every 16 steps for KettleBoiling (its
points are 80 apart), every 32 steps for SteamInMicrowave and PackIdenticalLunches (160 and 320 apart; refined to 16
around the jump afterwards). Each new point is a collect.py job with N = 8 candidates and M = 1 continuation, i.e.
8 independent plain rollouts.
Writes ../20261026_critic_data/jobs/dense_<task>.json and dense_intervals.json (here, beside the jobs it describes).
usage: python -I make_dense_jobs.py [--dry]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATA, HERE, JOBS, TASKS  # noqa: E402
from effect_by_event import points  # noqa: E402

JUMP, RISE_FROM = 0.4, 0.6
STEP = {"KettleBoiling": 16, "SteamInMicrowave": 32, "PackIdenticalLunches": 32}


def intervals():
    ev = {r["tag"]: r for r in map(json.loads, open(os.path.join(HERE, "events.jsonl")))}
    by = {}
    for p in points():
        by.setdefault(p["tag"], []).append(p)
    out = []
    for tag, g in by.items():
        g.sort(key=lambda p: p["t"])
        for a, b in zip(g[:-1], g[1:]):
            if a["k"] == b["k"] and abs(b["pd"] - a["pd"]) >= JUMP:
                out.append(dict(task=a["task"], tag=tag, k=a["k"], t0=a["t"], t1=b["t"], v0=a["pd"], v1=b["pd"], kind="jump"))
        for tc, k in ev[tag]["history"]:
            before = [p for p in g if p["k"] == k and p["t"] < tc]
            if before and before[-1]["pd"] <= RISE_FROM:
                a = before[-1]
                out.append(dict(task=a["task"], tag=tag, k=k, t0=a["t"], t1=tc, v0=a["pd"], v1=1.0, kind="rise_to_done"))
    return out


def main():
    dry = "--dry" in sys.argv
    iv = intervals()
    base = {t: {(j["tag"], j["t"]): j for j in json.load(open(os.path.join(DATA, f)))} for t, f in JOBS.items()}
    for task in TASKS:
        g = [i for i in iv if i["task"] == task]
        jobs = []
        for i in g:
            tmpl = next(j for (tag, _), j in base[task].items() if tag == i["tag"] and j["k"] == i["k"])
            for t in range(i["t0"] + STEP[task], i["t1"], STEP[task]):
                jobs.append(dict(tmpl, t=t, N=8, M=1))
        jobs = list({(j["tag"], j["t"]): j for j in jobs}.values())
        roll_steps = sum(j["L"] for j in jobs) * 8  # upper bound: every rollout runs its full budget
        print(f"{task:21s} intervals {len(g):3d} (jumps {sum(i['kind'] == 'jump' for i in g)}, drops {sum(i['kind'] == 'jump' and i['v1'] < i['v0'] for i in g)}, "
              f"rises to done {sum(i['kind'] == 'rise_to_done' for i in g)}) -> {len(jobs):4d} new points, {8 * len(jobs)} rollouts, "
              f"<= {roll_steps / 1e6:.1f} M steps")
        if not dry:
            json.dump(jobs, open(os.path.join(DATA, "jobs", f"dense_{task}.json"), "w"), indent=0)
    if not dry:
        json.dump(iv, open(os.path.join(HERE, "dense_intervals.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
