"""Phase 0: plain GR00T on one envkit scene (the paired baseline and the source of failure points).

The same seeding as envkit's policy test (tag "<task>_s<seed>", policy seed 0), so seeds 0-2 repeat those runs. Per
chunk it keeps the stage tracker and the oracle events (events.py; only the tasks in events.TARGETS have them).
Output: phase0/records/<task>_s<seed>.pkl.gz (envkit record + tracks + events), one json line on stdout.
usage: python -I plain.py <task> <seed>   (port from POLICY_PORT, default 5581)"""
import gzip, json, os, pickle, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import envkit, events, policy, stages  # noqa: E402

OUT = os.path.join(HERE, "phase0", "records")

if __name__ == "__main__":
    task, seed = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    try:
        client = policy.Client(int(os.environ.get("POLICY_PORT", 5581)))
        ep = envkit.Episode(task, seed)
        tracks, ev = [stages.track(ep)], events.Events(ep)
        evs = [[]]

        def on_chunk(e):
            tracks.append(stages.track(e))
            evs.append(ev.update(e))

        policy.rollout(ep, client, tag=f"{task}_s{seed}", on_chunk=on_chunk)
        rec = ep.record()
        rec.update(tracks=tracks, events=evs)
        ep.close()
        os.makedirs(OUT, exist_ok=True)
        with gzip.open(os.path.join(OUT, f"{task}_s{seed}.pkl.gz"), "wb") as f:
            pickle.dump(rec, f)
        first_done = {}
        for tr in tracks:
            for k, d in enumerate(tr["done"]):
                if d and k not in first_done:
                    first_done[k] = tr["t"]
        out = dict(task=task, seed=seed, success=rec["success"], success_step=rec["success_step"], steps=rec["steps"],
                   chunks=len(rec["chunks"]), n_stages=len(tracks[-1]["done"]), final_stage=tracks[-1]["stage"],
                   max_stage=max(tr["stage"] for tr in tracks), first_done=first_done,
                   events=[x for xs in evs for x in xs][:20],
                   seconds=round(time.time() - t0, 1))
    except Exception as x:
        import traceback
        out = dict(task=task, seed=seed, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-800:])
    print(json.dumps(out, default=str), flush=True)
