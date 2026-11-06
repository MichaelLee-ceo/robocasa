"""Offline evaluation of the VLM tracker on recorded GR00T episodes (2026-11-06).

The phase 0 plain runs (runs/20261105_recovery/phase0/records: 90 episodes of 3 tasks, GR00T alone, 39 successes) are
replayed exactly from their recorded chunks. The coordinator runs around each one with the VLM tracker and no detector
or recovery: it only watches, but its memory is built from the VLM's own verdicts, as in a live run (a sub-goal the
VLM calls done is closed, the next one begins). At every VLM check the oracle's view at the same chunk end (the
record's stages.track) is logged next to the answer, for scoring (score_tracker.py).

Context arms: what the memory adds to the request. The recent frames are the same in every arm (4 chunk ends, 2
chunks apart, 3 cameras: the pilot's clip), so the arms differ only in memory content.
  frames   the pilot's query: task, sub-goals, frames; no memory
  state    + the memory's progress state (each sub-goal: done at / CURRENT with its time vs. its usual time)
  text     + the tracker's previous verdict and evidence, the robot's own sensors, the notes kept so far
  full     + keyframe images (when the current sub-goal began, when the previous one was done)

usage: python -I coordinator/scripts/eval_tracker.py --vlm SPEC --arm ARM --out DIR [--every 4] TASK_sSEED ...
       writes DIR/<run>.json (every check with the oracle's view, and the memory at the end); a summary line per run"""
import argparse, gzip, json, os, pickle, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from coordinator import context as C, paths  # noqa: E402
from coordinator.detection.confirm import AcceptAll  # noqa: E402
from coordinator.detection.timeout import StageTimeout  # noqa: E402
from coordinator.loop import Coordinator  # noqa: E402
from coordinator.recovery.policies import RetraceRetry  # noqa: E402
from coordinator.tracking.oracle import task_plan  # noqa: E402
from coordinator.tracking.vlm import VLMTracker  # noqa: E402
from coordinator.vlm import make_vlm  # noqa: E402

paths.use_envkit()
import envkit  # noqa: E402

ARMS = {
    "frames": dict(progress_state=False, last_check=False, robot_state=False, notes=False, keyframes=False),
    "state": dict(progress_state=True, last_check=False, robot_state=False, notes=False, keyframes=False),
    "text": dict(progress_state=True, last_check=True, robot_state=True, notes=True, keyframes=False),
    "full": dict(progress_state=True, last_check=True, robot_state=True, notes=True, keyframes=True),
}
RECENT = dict(n_recent=4, recent_stride=2)


class EndOfRecord(Exception):
    pass


class Recorded:
    """GR00T replaced by the episode's recorded chunks"""

    def __init__(self, chunks):
        self.chunks = chunks

    def act(self, ep_obs, memory, chunk_index):
        if chunk_index >= len(self.chunks):
            raise EndOfRecord
        return self.chunks[chunk_index]["actions"]


class Shadow:
    """the VLM tracker, with the oracle's view at the same chunk end logged next to each answer (not shown to it)"""
    uses_sim = False

    def __init__(self, inner, tracks, log):
        self.inner, self.tracks, self.log = inner, tracks, log

    def make_plan(self, instruction, obs):
        return self.inner.make_plan(instruction, obs)

    def update(self, memory, obs):
        before, n_calls = memory.current, len(memory.vlm_calls)
        st = self.inner.update(memory, obs)
        call = memory.vlm_calls[-1] if len(memory.vlm_calls) > n_calls else {}
        tr = self.tracks[obs.chunk]
        self.log.append(dict(chunk=obs.chunk, t=obs.t, oracle_done=list(tr["done"]), oracle_stage=tr["stage"],
                             memory_before=before, done=st.done, current=st.current, error=st.error,
                             answer=call.get("answer"), seconds=call.get("seconds"),
                             input_tokens=call.get("input_tokens"), output_tokens=call.get("output_tokens"),
                             n_images=call.get("n_images")))
        return st


def run(run_id, vlm, arm, every, out_dir):
    task, seed = run_id.rsplit("_s", 1)
    rec = pickle.load(gzip.open(os.path.join(paths.RECORDS, f"{task}_s{seed}.pkl.gz")))
    ep = envkit.Episode(task, int(seed))
    ctx = C.ContextConfig(**ARMS[arm], **RECENT)
    log = []
    tracker = Shadow(VLMTracker(vlm, ctx, plan=task_plan(ep)), rec["tracks"], log)
    timeout = StageTimeout(task)
    expected = {k: timeout.window(k) for k in range(len(task_plan(ep)))}
    co = Coordinator(ep, Recorded(rec["chunks"]), tracker, [], AcceptAll(), RetraceRetry(), envkit.get_task_horizon(task),
                     free_retrace=False, track_every=every, expected=expected)
    t0 = time.time()
    co.start()
    try:
        co.run()
    except EndOfRecord:
        pass
    m = co.memory
    # the coordinator stops at the first success; phase 0 ran on to the horizon: compare the success step, else the length
    same = (bool(ep.success) and ep.success_step == rec["success_step"]) if rec["success"] else \
        (len(ep.chunks) == len(rec["chunks"]) and not ep.success)
    out = dict(run=run_id, task=task, seed=int(seed), arm=arm, vlm=getattr(vlm, "name", str(vlm)), every=every,
               context=ARMS[arm] | RECENT, record_success=rec["success"], record_chunks=len(rec["chunks"]),
               record_success_step=rec["success_step"], success_step=ep.success_step, chunks=len(ep.chunks),
               replay_matches_record=same, checks=log, memory=m.summary(), seconds=round(time.time() - t0, 1))
    ep.close()
    json.dump(out, open(os.path.join(out_dir, f"{run_id}.json"), "w"), default=str)
    n = len(log)
    ok = [c for c in log if c["done"] is not None]
    acc = sum(a == b for c in ok for a, b in zip(c["done"], c["oracle_done"])) / max(1, sum(len(c["done"]) for c in ok))
    return dict(run=run_id, arm=arm, checks=n, errors=n - len(ok), flag_acc=round(acc, 3), replay_ok=same,
                vlm_s=round(sum(c["seconds"] or 0 for c in log), 1), seconds=out["seconds"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vlm = make_vlm(a.vlm, structured=False, max_tokens=a.max_tokens) if not a.vlm.startswith(("claude", "mock")) \
        else make_vlm(a.vlm)
    for r in a.runs:
        if os.path.exists(os.path.join(a.out, f"{r}.json")):
            continue
        try:
            print(json.dumps(run(r, vlm, a.arm, a.every, a.out)), flush=True)
        except Exception as x:
            import traceback
            print(json.dumps(dict(run=r, arm=a.arm, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-1200:])),
                  flush=True)
