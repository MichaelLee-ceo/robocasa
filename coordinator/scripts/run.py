"""Run the coordinator on one episode.

  python -I coordinator/scripts/run.py point   <point id> <config> <policy seed> [--vlm SPEC] [--out DIR]
      from a phase 0 failure point: exact replay to it, then the failed sub-goal is stuck at once (detection assumed
      solved there); runs/20261105_recovery/phase0/points.json
  python -I coordinator/scripts/run.py episode <task> <scene seed> <config> <policy seed> [--vlm SPEC] [--out DIR]
      end to end, from the episode start
Options: --vlm a VLM spec (coordinator/vlm/__init__.py) for the VLM configs; --track-every N; --budget N (interventions
per episode); --plan given|vlm (the checklist of a VLM tracker). GR00T server port: POLICY_PORT (envkit/servers.sh).
Output: <out>/<config>/<run>_p<seed>.json (result, memory summary, uses_sim), <run>_p<seed>.pkl.gz (every env
chunk, as loop.py saved them; point runs) and one JSON line on stdout."""
import argparse, gzip, json, os, pickle, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from coordinator import configs, paths
from coordinator.points import load_point, run_from_point

paths.use_envkit()
import envkit, policy  # noqa: E402

FREE_HORIZON = 10 ** 6


def build(config, ep, port, tag, a):
    return configs.build(config, ep, port, tag, vlm=a.vlm, budget=a.budget, track_every=a.track_every, plan=a.plan)


def run_point(pid, config, pseed, port, a):
    point, rec = load_point(pid)
    ep = envkit.Episode(point["task"], point["seed"], horizon=FREE_HORIZON)
    co = build(config, ep, port, f"loop/{pid}/p{pseed}", a)   # loop.py's seed namespace, shared by every config
    out, _ = run_from_point(co, point, rec)
    out["_chunks"] = ep.chunks                                  # saved like loop.py's records (actions, success per step)
    ep.close()
    return out


def run_episode(task, seed, config, pseed, port, a):
    ep = envkit.Episode(task, seed, horizon=FREE_HORIZON if config != "plain" else None)
    if config == "plain":
        policy.rollout(ep, policy.Client(port), tag=f"coord/plain/{task}_s{seed}", policy_seed=pseed)
        out = dict(success=bool(ep.success), success_t=ep.success_step, chunks=len(ep.chunks))
    else:
        co = build(config, ep, port, f"episode/{task}_s{seed}/p{pseed}", a)
        co.start()
        out = co.run()
    out.update(task=task, seed=seed)
    ep.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["point", "episode"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--out", default=paths.OUT)
    ap.add_argument("--track-every", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--plan", default="given", choices=["given", "vlm"])
    a = ap.parse_args()
    t0, port = time.time(), int(os.environ.get("POLICY_PORT", 5581))
    if a.mode == "point":
        rid, config, pseed = a.args[0], a.args[1], int(a.args[2])
    else:
        rid, config, pseed = f"{a.args[0]}_s{a.args[1]}", a.args[2], int(a.args[3])
    try:
        r = run_point(rid, config, pseed, port, a) if a.mode == "point" else \
            run_episode(a.args[0], int(a.args[1]), config, pseed, port, a)
    except Exception as x:
        import traceback
        r = dict(error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-1500:])
    r.update(run=rid, config=config, pseed=pseed, vlm=a.vlm, seconds=round(time.time() - t0, 1))
    d = os.path.join(a.out, config)
    os.makedirs(d, exist_ok=True)
    chunks = r.pop("_chunks", None)
    if chunks is not None:   # every env chunk (GR00T's and the retraces'): actions, success per step, as loop.py saved
        with gzip.open(os.path.join(d, f"{rid}_p{pseed}.pkl.gz"), "wb") as f:
            pickle.dump(dict(out={k: v for k, v in r.items() if k != "memory"}, chunks=chunks), f)
    json.dump(r, open(os.path.join(d, f"{rid}_p{pseed}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in r.items() if k != "memory"}, default=str), flush=True)
