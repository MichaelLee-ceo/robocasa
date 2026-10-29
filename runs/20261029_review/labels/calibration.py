"""Global restore-fidelity test: V(t) against the main run's own outcome from the same state.

At a decision point, the main run's continuation from its state at t is one more draw of the same process as the
rollouts (same observation, same policy, different noise), if the restore is faithful. So Y_main(t) = 1 if the main
run finished stage k within L steps after t (history entry for k at tc <= t + L; for the last stage also its
success at steps <= t + L), 0 if it did not although its episode ran at least to t + L; censored otherwise.
E[Y_main(t)] = E[V(t)] under faithful restore. A restore that perturbs the rollouts (bugs 1-4) makes V < Y_main on
average. Points of one (episode, stage) share one main-run trajectory, so CIs come from a cluster bootstrap over
(episode, stage). Breakdowns by task, stage, code version (hidden), source, and V bin.
Writes calibration.json. usage: python calibration.py"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, NSTAGES, TASKS3, main_runs  # noqa: E402
from vseries import build  # noqa: E402

L = {"KettleBoiling": [416, 272], "SteamInMicrowave": [432, 720, 576, 416], "PackIdenticalLunches": [416, 576, 608, 496]}
rng = np.random.default_rng(0)


def y_main(mr, p):
    r = mr[p["tag"]]
    Lk = L[p["task"]][p["k"]]
    tcs = [tc for tc, k in r["history"] if k == p["k"]]
    last = p["k"] == NSTAGES[p["task"]] - 1
    if tcs and tcs[0] - p["t"] <= Lk:
        return 1
    if last and r["success"] and r["steps"] - p["t"] <= Lk:
        return 1
    if r["steps"] >= p["t"] + Lk:  # ran the whole window without finishing
        return 0
    if tcs or (last and r["success"]):
        return 0  # finished, but later than t + L
    return None  # episode ended (horizon) before t + L: censored


def boot(g):
    """cluster bootstrap of mean(Y) - mean(V) over (tag, k) clusters"""
    cl = {}
    for p in g:
        cl.setdefault((p["tag"], p["k"]), []).append(p)
    keys = list(cl)
    diffs = []
    for _ in range(2000):
        s = [cl[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        pts = [p for c in s for p in c]
        diffs.append(np.mean([p["y"] for p in pts]) - np.mean([p["V"] for p in pts]))
    return np.percentile(diffs, [2.5, 97.5])


def line(lab, g):
    if len(g) < 5:
        return f"   {lab:42s} n={len(g)}"
    lo, hi = boot(g)
    return (f"   {lab:42s} n={len(g):4d} clusters={len({(p['tag'], p['k']) for p in g}):3d}  mean V={np.mean([p['V'] for p in g]):.3f}  "
            f"main-run rate={np.mean([p['y'] for p in g]):.3f}  diff {np.mean([p['y'] for p in g]) - np.mean([p['V'] for p in g]):+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main():
    mr = main_runs()
    pts, _ = build()
    for p in pts:
        p["y"] = y_main(mr, p)
    use = [p for p in pts if p["y"] is not None]
    print(f"{len(pts)} points, {len(use)} with an uncensored main-run outcome")
    out = {}
    for task in TASKS3:
        g = [p for p in use if p["task"] == task]
        print(task)
        print(line("all", g))
        for k in range(NSTAGES[task]):
            print(line(f"S{k}", [p for p in g if p["k"] == k]))
        for h in sorted({str(p["hidden"]) for p in g}):
            print(line(f"hidden={h}", [p for p in g if str(p["hidden"]) == h]))
        for src in ("coarse", "dense"):
            print(line(f"src={src}", [p for p in g if p["src"] == src]))
        out[task] = [dict(tag=p["tag"], t=p["t"], k=p["k"], V=p["V"], y=p["y"], hidden=p["hidden"], src=p["src"]) for p in g]
    print("all tasks, by V bin (calibration curve)")
    for a, b in ((0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        print(line(f"V in [{a}, {b})", [p for p in use if a <= p["V"] < b]))
    print(line("all tasks, all points", use))
    json.dump(out, open(os.path.join(HERE, "calibration.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
