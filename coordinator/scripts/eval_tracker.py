"""Offline evaluation of the VLM tracker (the to-do list, tracking/vlm.py) on recorded GR00T episodes (2026-11-07).

The phase 0 plain runs (runs/20261105_recovery/phase0/records: 90 episodes of 3 tasks, GR00T alone, 39 successes) are
replayed exactly from their recorded chunks. The coordinator runs around each one with the VLM tracker and no detector
or recovery: it only watches, but its memory (the to-do list) is built from the VLM's own answers, as in a live run:
an item it ticks is closed and the next one begins. At every check the oracle's view at the same chunk end (the
record's stages.track) is logged next to every VLM call made at that check, for scoring (score_tracker.py). The memory
at the end holds the plan, the progress dots and the ticks.

Arms (--window): which moments the VLM sees; everything else is the same (the to-do list text, the robot's sensors,
the notes, the three cameras, the questions):
  uniform  the user's design (2026-11-08): 10 chunk ends sampled uniformly from the start of the current attempt (the
           item's start; after a retry, the end of the rewind) to now
  recent   a fixed window: the last 10 chunk ends, 2 chunks apart, ending now
(Earlier batches: runs/20261107_todo_offline had "attempt" (the previous item's completion, the attempt's start, 4
spread, 2 recent, now; two cameras) and a hand-off check before each tick; runs/20261108_todo_uniform had this
uniform window with the hand-off check (stopped at 15/90 when the check was dropped). See their READMEs.)
--plan vlm (default): the VLM writes the to-do list from the task instruction; given: the task's stage sentences
(tracking.oracle.task_plan, no done conditions).

usage: python -I coordinator/scripts/eval_tracker.py --vlm SPEC --window uniform|recent --out DIR [--plan vlm|given]
       [--every 1] [--max-chunks N] TASK_sSEED ...
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

    @property
    def plan_call(self):            # the coordinator logs the VLM's plan call from here
        return self.inner.plan_call

    def update(self, memory, obs):
        before, n_calls = memory.current, len(memory.vlm_calls)
        attempt = len(memory.stages[before].attempts) if before < len(memory.stages) else None
        n_done = sum(memory.is_done(i) for i in range(len(memory.stages)))
        st = self.inner.update(memory, obs)
        calls = [dict(module=c["module"], answer=c.get("answer"), error=c.get("error"), seconds=c.get("seconds"),
                      input_tokens=c.get("input_tokens"), output_tokens=c.get("output_tokens"),
                      n_images=c.get("n_images")) for c in memory.vlm_calls[n_calls:]]
        tr = self.tracks[obs.chunk]
        self.log.append(dict(chunk=obs.chunk, t=obs.t, oracle_done=list(tr["done"]), oracle_stage=tr["stage"],
                             memory_before=before, attempt=attempt, n_done=n_done, done=st.done, current=st.current,
                             extra_done=st.extra_done, error=st.error, calls=calls))
        return st


def run(run_id, vlm, window, plan, every, out_dir, max_chunks=None):
    task, seed = run_id.rsplit("_s", 1)
    rec = pickle.load(gzip.open(os.path.join(paths.RECORDS, f"{task}_s{seed}.pkl.gz")))
    ep = envkit.Episode(task, int(seed))
    ctx = C.ContextConfig(window=window)
    log = []
    tracker = Shadow(VLMTracker(vlm, ctx, plan=task_plan(ep) if plan == "given" else None), rec["tracks"], log)
    timeout = StageTimeout(task)
    expected = {k: timeout.window(k) for k in range(len(task_plan(ep)))}
    co = Coordinator(ep, Recorded(rec["chunks"][:max_chunks]), tracker, [], AcceptAll(), RetraceRetry(), envkit.get_task_horizon(task),
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
    out = dict(run=run_id, task=task, seed=int(seed), window=window, plan_source=plan, vlm=getattr(vlm, "name", str(vlm)),
               every=every, context=dict(vars(ctx)), oracle_plan=[g.text for g in task_plan(ep)],
               record_success=rec["success"], record_chunks=len(rec["chunks"]),
               record_success_step=rec["success_step"], success_step=ep.success_step, chunks=len(ep.chunks),
               replay_matches_record=same, checks=log, memory=m.summary(), seconds=round(time.time() - t0, 1))
    ep.close()
    json.dump(out, open(os.path.join(out_dir, f"{run_id}.json"), "w"), default=str)
    n = len(log)
    ok = [c for c in log if c["done"] is not None]
    return dict(run=run_id, window=window, plan=len(m.plan), oracle_plan=len(out["oracle_plan"]), checks=n,
                errors=n - len(ok), ticks=len(m.moved_on), replay_ok=same,
                vlm_s=round(sum(x["seconds"] or 0 for c in log for x in c["calls"]), 1), seconds=out["seconds"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--window", required=True, choices=["uniform", "recent"])
    ap.add_argument("--plan", default="vlm", choices=["vlm", "given"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max-chunks", type=int, default=None, help="smoke tests: replay only the first N chunks")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vlm = make_vlm(a.vlm, structured=False, max_tokens=a.max_tokens) if not a.vlm.startswith(("claude", "mock")) \
        else make_vlm(a.vlm)
    for r in a.runs:
        if os.path.exists(os.path.join(a.out, f"{r}.json")):
            continue
        try:
            print(json.dumps(run(r, vlm, a.window, a.plan, a.every, a.out, a.max_chunks)), flush=True)
        except Exception as x:
            import traceback
            print(json.dumps(dict(run=r, window=a.window, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-1200:])),
                  flush=True)
