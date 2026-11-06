"""State or scene? From a failure point's scene, start again at the moment its failed stage began (phase 0 replayed
exactly up to the stage start, before anything went wrong) and let GR00T run with fresh samples, the task sentence,
for the stage's stall window (the STALL_Q percentile of its completions in phase 0, the window that defined the
failure). If the stage gets done much more often from here than from the failure point (loop.py control arm, the same
window), the failure state is the problem and the loop should go back to the stage start; if both are low, the stage
is hard in this scene and retries alone will not fix it.
Output: stage_start/<point id>_p<pseed>.json; one json line on stdout.
usage: python -I stage_start.py <point id> <policy seed>   (port from POLICY_PORT)"""
import gzip, json, os, pickle, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import envkit, policy, stages  # noqa: E402
from loop import stall_limits  # noqa: E402


def run(pid, pseed, port):
    t0 = time.time()
    point = next(p for p in json.load(open(os.path.join(HERE, "phase0", "points.json"))) if p["id"] == pid)
    task, seed, s = point["task"], point["seed"], point["stage"]
    rec = pickle.load(gzip.open(os.path.join(HERE, "phase0", "records", f"{task}_s{seed}.pkl.gz")))
    window = stall_limits().get((task, s), 40)
    client = policy.Client(port)
    ep = envkit.Episode(task, seed)
    for c in rec["chunks"][:point["stage_start"]]:  # tracks[i] = after i chunks: the stage is current from here
        ep.step(c["actions"])
    tr = stages.track(ep)
    assert tr["stage"] == s, f"{pid}: stage {tr['stage']} at the stage start, expected {s}"
    start, done_after = len(ep.chunks), None
    tag = f"stage_start/{pid}/p{pseed}"
    while not ep.done and len(ep.chunks) - start < window:
        ep.step(policy.act(client, ep.obs, policy.chunk_seed(tag, 0, len(ep.chunks))))
        tr = stages.track(ep)
        if tr["done"][s] or ep.success:
            done_after = len(ep.chunks) - start
            break
    out = dict(id=pid, task=task, seed=seed, stage=s, pseed=pseed, stage_start=start,
               window=window, chunks_left=len(rec["chunks"]) - start, done=done_after is not None,
               done_after=done_after, seconds=round(time.time() - t0, 1))
    d = os.path.join(HERE, "stage_start")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{pid}_p{pseed}.json"), "w"), indent=1)
    with gzip.open(os.path.join(d, f"{pid}_p{pseed}.pkl.gz"), "wb") as f:
        pickle.dump(dict(out=out, start_chunk=start, chunks=ep.chunks), f)
    ep.close()
    return out


if __name__ == "__main__":
    pid, pseed = sys.argv[1], int(sys.argv[2])
    try:
        r = run(pid, pseed, int(os.environ.get("POLICY_PORT", 5581)))
    except Exception as x:
        import traceback
        r = dict(id=pid, pseed=pseed, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-800:])
    print(json.dumps(r, default=str), flush=True)
