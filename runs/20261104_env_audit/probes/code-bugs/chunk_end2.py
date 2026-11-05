"""per task (one worker): successful seen480 episodes; is the official check True at the step where the rollout stopped
(the last step of the chunk in which the reward first fired)? GR00T simulation.py counts info['success'] of the last step
of each chunk only. Latching tasks are replayed in full, the others only over the last chunk."""
import sys, json, glob, multiprocessing as mp
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480"):
    sys.path.insert(0, f"{R}/{p}")
FULL = {"RinseSinkBasin", "GetToastedBread", "WashLettuce", "ScrubCuttingBoard", "StirVegetables"}


def work(task):
    from analyze_stages import replay_env
    import atomic_eval  # noqa
    from atomic_eval import load_scene
    from oracle import set_state
    from paths import local
    rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
    rows = [r for r in rows if r["task"] == task and r.get("success")]
    env = replay_env(task)
    out = []
    for row in rows:
        states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{row['tag']}_steps.npz")["states"]
        load_scene(env, local(row["scene"]))
        lo = 0 if task in FULL else max(1, len(states) - 17)
        succ = {}
        for i in range(lo, len(states)):
            set_state(env, states[i])
            succ[i] = bool(env._check_success()) if i else False
        last = len(states) - 1
        tail = [i for i in range(max(1, last - 15), last + 1)]
        o = dict(tag=row["tag"], steps=row["steps"], n_states=len(states), at_stop=succ[last],
                 in_last_chunk=sum(succ[i] for i in tail))
        out.append(o)
        with open(f"chunk_end2_{task}.jsonl", "a") as f:
            f.write(json.dumps(o) + "\n")
    return task, out


if __name__ == "__main__":
    tasks = sys.argv[1:]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        for task, out in pool.imap_unordered(work, tasks):
            bad = [o for o in out if not o["at_stop"]]
            print(f"{task:22s} successes {len(out):3d}  check False at the stop step: {len(bad)} {[(o['tag'], o['in_last_chunk']) for o in bad]}", flush=True)
