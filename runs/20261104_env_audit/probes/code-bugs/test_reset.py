"""Does load_scene reset ALL python state (env attributes + every fixture's plain attributes) for every task?
fresh env: load_scene -> snapshot A. Then replay a whole seen480 episode (set_state + _check_success per step, as the
live run does), load_scene the same scene again -> snapshot B. Report what differs."""
import sys, json, glob, copy, multiprocessing as mp
import numpy as np
R = "/home/michaellee/mclee/robocasa"
for p in ("runs/scripts", "runs/20261025_seen480"):
    sys.path.insert(0, f"{R}/{p}")


def plain(v):
    if v is None or isinstance(v, (bool, int, float, str, np.bool_, np.integer, np.floating)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype != object and v.size < 10000
    if isinstance(v, (list, tuple)):
        return all(plain(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, (str, int)) and plain(x) for k, x in v.items())
    return False


def snap(env):
    import hidden_state
    out = {"env/" + k: copy.deepcopy(v) for k, v in vars(env).items() if plain(v) and not k.startswith("_obs_cache")}
    for n, fx in env.fixtures.items():
        for k, v in vars(fx).items():
            if plain(v):
                out[f"fx/{n}/{k}"] = copy.deepcopy(v)
    return out


def eq(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(np.asarray(a), np.asarray(b))
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
        return a == b
    except Exception:
        return False


def work(task):
    try:
        from analyze_stages import replay_env
        import atomic_eval  # noqa
        from atomic_eval import load_scene
        from oracle import set_state
        from paths import local
        rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
        rs = sorted([r for r in rows if r["task"] == task], key=lambda r: (not r["success"], -r["steps"]))
        row = rs[0]
        states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{row['tag']}_steps.npz")["states"]
        env = replay_env(task)
        load_scene(env, local(row["scene"]))
        A = snap(env)
        for i in range(len(states)):
            set_state(env, states[i]); env._check_success() if i else None
        mid = snap(env)
        load_scene(env, local(row["scene"]))
        B = snap(env)
        changed_by_episode = sorted(k for k in set(A) | set(B) if k in A and k in mid and not eq(A[k], mid[k]))
        diff = sorted(k for k in set(A) | set(B) if (k in A) != (k in B) or not eq(A.get(k), B.get(k)))
        return dict(task=task, tag=row["tag"], success=row["success"], changed_by_episode=changed_by_episode,
                    not_reset_by_load_scene={k: (str(A.get(k))[:80], str(B.get(k))[:80]) for k in diff})
    except Exception as e:
        import traceback
        return dict(task=task, error=repr(e), tb=traceback.format_exc()[-800:])


if __name__ == "__main__":
    tasks = ["KettleBoiling", "PreSoakPan", "StackBowlsCabinet", "SteamInMicrowave", "PrepareCoffee", "ScrubCuttingBoard",
             "WashLettuce", "RinseSinkBasin", "LoadDishwasher", "StoreLeftoversInBowl", "SetUpCuttingStation",
             "StirVegetables", "SearingMeat", "PackIdenticalLunches", "DeliverStraw", "GetToastedBread"]
    with mp.get_context("spawn").Pool(16) as pool:
        out = pool.map(work, tasks, chunksize=1)
    json.dump(out, open("test_reset.json", "w"), default=str, indent=1)
    for o in out:
        print(json.dumps(o, default=str)[:1500])
