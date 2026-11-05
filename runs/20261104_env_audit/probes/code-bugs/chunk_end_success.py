"""successful seen480 episodes: does the official check also hold at the chunk end where the rollout stopped
(GR00T's simulation.py reads info['success'] of the last step of each chunk only)?"""
import sys, json, glob, multiprocessing as mp
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480"):
    sys.path.insert(0, f"{R}/{p}")


def work(row):
    from analyze_stages import replay_env
    import atomic_eval  # noqa
    from atomic_eval import load_scene
    from oracle import set_state
    from paths import local
    task = row["task"]
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{row['tag']}_steps.npz")["states"]
    env = replay_env(task)
    load_scene(env, local(row["scene"]))
    succ = []
    for i in range(len(states)):
        set_state(env, states[i])
        succ.append(bool(env._check_success()) if i else False)
    first = succ.index(True) if any(succ) else None
    ends = [i for i in range(16, len(states), 16)] + [len(states) - 1]
    return dict(tag=row["tag"], task=task, steps=row["steps"], n=len(states), first=first, at_last=succ[-1],
                any_end=any(succ[i] for i in ends), n_true=sum(succ))


if __name__ == "__main__":
    rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
    rows = [r for r in rows if r.get("success") and "error" not in r]
    with mp.get_context("spawn").Pool(24) as pool:
        out = pool.map(work, rows, chunksize=1)
    json.dump(out, open("chunk_end_success.json", "w"))
    import collections
    c = collections.defaultdict(lambda: [0, 0, 0])
    for o in out:
        c[o["task"]][0] += 1; c[o["task"]][1] += (not o["at_last"]); c[o["task"]][2] += (o["first"] is None)
    for t, v in sorted(c.items()):
        print(f"{t:22s} successes {v[0]:3d}  official check False at the stop step {v[1]:2d}  never True in replay {v[2]}")
    for o in out:
        if not o["at_last"] or o["first"] is None:
            print(o)
