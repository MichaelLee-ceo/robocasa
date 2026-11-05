"""collect.py replay() vs the main run for _check_success latches (RinseSinkBasin washed_loc, GetToastedBread toaster_on)"""
import sys, json, glob, multiprocessing as mp
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480"):
    sys.path.insert(0, f"{R}/{p}")
ATTR = {"RinseSinkBasin": "washed_loc", "GetToastedBread": "toaster_on"}


def work(row):
    from analyze_stages import replay_env
    import atomic_eval  # noqa
    from atomic_eval import load_scene
    from oracle import set_state
    from atomic_steps import TASKS
    from paths import local
    import copy
    task = row["task"]; steps = TASKS[task]; a = ATTR[task]
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{row['tag']}_steps.npz")["states"]
    env = replay_env(task)
    load_scene(env, local(row["scene"]))
    faith = []
    for i in range(len(states)):
        set_state(env, states[i]); env._check_success() if i else None
        faith.append(copy.deepcopy(getattr(env, a)))
    load_scene(env, local(row["scene"]))
    k, S, coll = 0, {}, []
    for i in range(len(states)):  # collect.py replay(): set_state per step, stage checks at chunk boundaries only
        set_state(env, states[i])
        if i and i % 16 == 0:
            while k < len(steps) and steps[k]["check"](env, S):
                k += 1
        coll.append(copy.deepcopy(getattr(env, a)))
    d = [i for i in range(0, len(states), 16) if faith[i] != coll[i]]
    return dict(tag=row["tag"], success=row["success"], n_bound=len(range(0, len(states), 16)), n_diff=len(d),
                first=(d[0], faith[d[0]], coll[d[0]]) if d else None)


if __name__ == "__main__":
    task = sys.argv[1]
    rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
    rows = [r for r in rows if r["task"] == task]
    with mp.get_context("spawn").Pool(15) as pool:
        out = pool.map(work, rows, chunksize=1)
    with open(f"scan_{task}.json", "w") as f:
        json.dump(out, f, default=str)
    for o in out:
        if o["n_diff"]:
            print(o)
    print(f"{task}: {sum(o['n_diff'] > 0 for o in out)}/{len(out)} episodes where replay()'s {ATTR[task]} differs from the main run's at some chunk boundary")
