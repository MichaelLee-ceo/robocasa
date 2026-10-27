"""Stage-level root cause of finished rerun episodes (runs/scripts/stage_trace.py + the official subtask labeler).

Per episode (env built like the rollout's but without rendering, see replay_env; scene restored with oracle.reset_to,
per-step states replayed):
  * stage_trace: every stage check per frame (order-free), and for each stage not complete at the end, where the
    manipulation broke (verdicts in stage_trace.py);
  * official labeler (label_episode from add_subtask_labels_to_lerobot.py): its segments, validate() issues and debug
    events, as context for what the robot was doing;
  * sanity: the order-free first frame of every stage that the rollout's tracker logged must be <= that logged frame
    (the rollout checks only at chunk boundaries, every 16 steps) and >= it - 16.
One jsonl line per episode in <out>/stages.jsonl (a rerun skips done tags); per-frame signals in
<out>/<task>/<tag>_trace.json.gz. usage: [TAGS=a,b,...] python -I analyze_stages.py <run dir> <out dir> <workers> [max episodes]"""
import glob, gzip, json, multiprocessing, os, sys
import numpy as np
import gymnasium as gym
R = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{R}/runs/scripts")


_ENV = {}


def replay_env(task):
    """The rollout's env (RoboCasaGymEnv -> robocasa create_env, split target), without any rendering: same robosuite.make
    arguments except has_offscreen_renderer / use_camera_obs = False, so the scene regeneration in reset_to matches the
    rollout while no worker opens an EGL context (32 rendering-enabled replay workers queued on the NVIDIA driver lock).
    One env per task per worker process, reused across episodes."""
    if task not in _ENV:
        import robosuite
        from robocasa.utils.env_utils import create_env
        for e in _ENV.values():
            e.close()
        _ENV.clear()
        make = robosuite.make
        robosuite.make = lambda **kw: make(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
        try:
            env = create_env(env_name=task, split="target", render_onscreen=False)
        finally:
            robosuite.make = make
        env.reset()
        _ENV[task] = env
    return _ENV[task]


def work(job):
    import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L
    from robocasa.utils.subtask_labeling import get_labeler
    import atomic_eval  # noqa: F401  (registers the robocasa gym envs, as in the rollout)
    from oracle import reset_to, set_state
    from stage_trace import trace_episode, analyze
    r, run, out = job
    task, tag = r["task"], r["tag"]
    try:
        z = np.load(r["scene"], allow_pickle=True)
        xml, ep_meta = str(z["xml"]), json.loads(str(z["ep_meta"]))
        states = np.load(f"{run}/{task}/{tag}_steps.npz")["states"]
        env = replay_env(task)
        reset_to(env, xml, ep_meta)
        rec = trace_episode(env, task, states, set_state)
        summ = analyze(rec, task)
        sanity = [dict(stage=k, logged=t, replay_first=summ["stages"][k]["first_true"],
                       ok=summ["stages"][k]["first_true"] is not None and t - 16 <= summ["stages"][k]["first_true"] <= t)
                  for t, k in r["history"]]
        labeler = get_labeler(task)
        lab = L.label_episode(env, labeler, states, ep_meta, xml)
        idx = np.asarray(lab.subtask_idx)
        starts = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
        segs = [[int(s), str(lab.subtask_name[s]), str(lab.subtask_stage[s])] for s in starts]
        os.makedirs(f"{out}/{task}", exist_ok=True)
        with gzip.open(f"{out}/{task}/{tag}_trace.json.gz", "wt") as f:
            json.dump(rec, f, default=lambda v: v.item() if hasattr(v, "item") else str(v))  # numpy scalars from fixture APIs
        return dict(tag=tag, task=task, success=r["success"], steps=r["steps"], history=r["history"], **summ,
                    sanity_ok=all(s["ok"] for s in sanity), sanity=sanity, labeler_segments=segs,
                    labeler_issues=labeler.validate(lab),
                    labeler_debug={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in lab.debug.items()})
    except Exception as e:
        import traceback
        return dict(tag=tag, task=task, error=repr(e), tb=traceback.format_exc()[-1500:])


if __name__ == "__main__":
    run, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(out, exist_ok=True)
    jl = f"{out}/stages.jsonl"
    done = {r["tag"] for r in map(json.loads, open(jl)) if "error" not in r} if os.path.exists(jl) else set()  # errors are retried
    rows = [json.loads(l) for p in sorted(glob.glob(f"{run}/results_w*.jsonl")) for l in open(p)]
    jobs = [(r, run, out) for r in rows if "error" not in r and r.get("n_step_states") and r["tag"] not in done]
    if os.environ.get("TAGS"):  # optional comma-separated subset of tags
        keep = set(os.environ["TAGS"].split(","))
        jobs = [j for j in jobs if j[0]["tag"] in keep]
    jobs = jobs[:cap] if cap else jobs
    print(f"{len(jobs)} episodes to analyze ({len(done)} already done), {workers} workers", flush=True)
    with multiprocessing.get_context("spawn").Pool(workers) as pool, open(jl, "a") as f:
        for res in pool.imap_unordered(work, jobs, chunksize=1):
            f.write(json.dumps(res, default=str) + "\n"); f.flush()
            first = res.get("first_incomplete_stage")
            v = None
            if first is not None:
                det = res["stages"][first].get("details", [{}])
                v = [d.get("verdict", d.get("kind")) for d in det]
            print(json.dumps(dict(tag=res["tag"], success=res.get("success"), first_incomplete=first, verdict=v,
                                  sanity=res.get("sanity_ok"), error=res.get("error"))), flush=True)
