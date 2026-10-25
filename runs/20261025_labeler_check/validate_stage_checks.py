"""Our per-stage checks (atomic_steps.TASKS) vs the official stage segmentation on official demos (step 2 of the
(c) validation). Generalises prev/_handoff_verify_asus/seen11/validate_seen11.py to all 16 Composite-Seen tasks.

Per demo: env from the dataset's env_args (official build_env), scene restored with oracle.reset_to (the reset our
rollouts use), then every frame: set_state (runs update_state), env._check_success() once (what env.step's reward()
does; some checks latch inside it), then every stage check evaluated independently (order-free, first frame each is
True) and the sequential tracker as in the rollouts. Official stage units come from the stored per-frame annotation
(identical to the official labeler's output, see check_official_labels.py) via atomic_steps_seen11.step_segments.
Separately, after its own reset_to, every check must be False at the start frame of its annotated unit.
usage: python -I validate_stage_checks.py <episodes per task> <workers> [comma-separated tasks]
Demos run in a process pool; each finished demo is appended to validate_stage_checks.jsonl (a rerun skips demos
already there), and validate_stage_checks.json (per-stage summary) is rebuilt from it at the end."""
import glob, json, multiprocessing, os, sys, time
from pathlib import Path
import numpy as np
R = os.environ["ROBOCASA_ROOT"]
sys.path.insert(0, f"{R}/runs/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script dir off sys.path
import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L  # noqa: E402
from demo_utils import load_demo, n_episodes  # noqa: E402
from oracle import reset_to, set_state  # noqa: E402
from atomic_steps import TASKS  # noqa: E402
from atomic_steps_seen11 import step_segments  # noqa: E402
from check_official_labels import SEEN  # noqa: E402


def validate_demo(env, task, ep):
    steps = TASKS[task]
    t0 = time.time()
    d = load_demo(task, ep)
    units = step_segments(d)
    rec = dict(task=task, ep=ep, length=d["length"], annotation_matches=[u["skill"] for u in units] == [s["skill"] for s in steps])
    false_at_start = []
    for i, s in enumerate(steps):
        if i < len(units):
            reset_to(env, d["xml"], d["ep_meta"], d["states"][units[i]["start"]])
            false_at_start.append(not bool(s["check"](env, {})))
    reset_to(env, d["xml"], d["ep_meta"])
    S, S_seq, cur, seq_fired = {}, {}, 0, []
    first = [None] * len(steps)
    for f in range(len(d["states"])):
        set_state(env, d["states"][f])
        env._check_success()
        for i, s in enumerate(steps):
            if first[i] is None and s["check"](env, S):
                first[i] = f
        while cur < len(steps) and steps[cur]["check"](env, S_seq):
            seq_fired.append(f); cur += 1
    ends = [u["end"] for u in units]
    rec.update(false_at_unit_start=false_at_start, first_pass=first, seq_first_pass=seq_fired, unit_end=ends,
               pass_minus_unit_end=[None if first[i] is None or i >= len(ends) else first[i] - ends[i] for i in range(len(steps))],
               final_success=bool(env._check_success()), wall_s=round(time.time() - t0, 1))
    return rec


_ENV = {}


def _work(job):
    task, ep = job
    if task not in _ENV:  # one env per task per worker, built from the dataset's env_args
        for e in _ENV.values():
            e.close()
        _ENV.clear()
        lerobot = Path(sorted(glob.glob(f"{R}/datasets/v1.0/target/composite/{task}/*/lerobot"))[-1])
        _ENV[task] = L.build_env(json.loads((lerobot / "extras" / "dataset_meta.json").read_text())["env_args"])
    try:
        return validate_demo(_ENV[task], task, ep)
    except Exception as e:  # keep going; the error is recorded
        return dict(task=task, ep=ep, error=repr(e))


def summarize(rows_by_task):
    report = {}
    for task, rows in rows_by_task.items():
        steps = TASKS[task]
        rows = sorted((r for r in rows if "error" not in r), key=lambda r: r["ep"])
        lag = [[r["pass_minus_unit_end"][i] for r in rows] for i in range(len(steps))]
        report[task] = dict(n=len(rows), episodes=rows, stages=[
            dict(stage=i, skill=steps[i]["skill"], never_passed=sum(x is None for x in lag[i]),
                 true_at_unit_start=sum(not r["false_at_unit_start"][i] for r in rows if i < len(r["false_at_unit_start"])),
                 lag_frames_p10_med_p90=(np.percentile([x for x in lag[i] if x is not None], [10, 50, 90]).round().tolist()
                                         if any(x is not None for x in lag[i]) else None)) for i in range(len(steps))])
    return report


if __name__ == "__main__":
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    tasks = sys.argv[3].split(",") if len(sys.argv) > 3 else SEEN
    here = Path(__file__).resolve().parent
    jl = here / "validate_stage_checks.jsonl"
    done = {(r["task"], r["ep"]) for r in map(json.loads, jl.read_text().splitlines()) if "error" not in r} if jl.exists() else set()
    jobs = [(t, ep) for t in tasks for ep in sorted({int(i * n_episodes(t) / n) for i in range(n)}) if (t, ep) not in done]
    print(f"{len(jobs)} demos to run ({len(done)} already done), {workers} workers", flush=True)
    with multiprocessing.get_context("spawn").Pool(workers) as pool, open(jl, "a") as f:
        for rec in pool.imap_unordered(_work, jobs, chunksize=1):
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
            print(json.dumps({k: rec.get(k) for k in ("task", "ep", "wall_s", "error")}), flush=True)
    rows = {}
    for r in map(json.loads, jl.read_text().splitlines()):
        rows.setdefault(r["task"], []).append(r)
    report = summarize(rows)
    (here / "validate_stage_checks.json").write_text(json.dumps(report, indent=1, default=str))
    for t, v in report.items():
        print(t, json.dumps(v["stages"]), flush=True)
