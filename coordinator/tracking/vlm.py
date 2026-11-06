"""The VLM tracker (the proposal's progress tracking): a VLM judges the checklist from the memory.

Reference version v0: it runs, but its prompt and thresholds are untuned and it has not been scored against the
oracle. Owner: ___. Before it runs in the loop at scale, score it on recorded runs against OracleTracker (precision /
recall of "done" per sub-goal; a false "done" is the dangerous error: the coordinator moves on and never retries).

make_plan: the given checklist (`plan`, e.g. tracking.oracle.task_plan: the task's stage sentences), or, with
    plan=None, the VLM splits the task sentence itself (context.plan_request).
update: context.progress_request (the pilot's validated question and answer format, coordinator/vlm/pilot.py, on the pilot
    Query, with the memory added: each sub-goal's state, the previous verdict, the robot's sensors, notes, the sub-goal's
    start and the previous completion, recent moments) -> StageStatus. An event other than "none" in the answer
    (missed_grasp, dropped, ...) is kept as a note. A sub-goal newly called done must be
    called done `done_streak` times in a row with confidence >= `min_confidence` before it counts. A failed call returns
    StageStatus(done=None): the coordinator keeps its current view."""
from .. import context as C
from ..schema import StageStatus, SubGoal
from ..vlm import make_vlm
from .base import ProgressTracker


class VLMTracker(ProgressTracker):
    def __init__(self, vlm, cfg=None, plan=None, done_streak=1, min_confidence=0.0):
        self.vlm, self.cfg, self.given_plan = make_vlm(vlm), cfg or C.ContextConfig(), plan
        self.done_streak, self.min_confidence = done_streak, min_confidence
        self.streak, self.plan_call = {}, None

    def make_plan(self, instruction, obs):
        if self.given_plan is not None:
            return list(self.given_plan)
        res = self.plan_call = self.vlm.ask(C.plan_request(instruction, obs, self.cfg))
        if not res.ok:
            raise RuntimeError(f"the VLM could not make the plan: {res.error}")
        return [SubGoal(i, g["text"], g.get("check", "")) for i, g in enumerate(res.data["subgoals"])]

    def update(self, memory, obs):
        res = self.vlm.ask(C.progress_request(memory, obs, self.cfg))
        memory.log_vlm("tracker", obs, res)
        K = len(memory.plan)
        if not res.ok:
            return StageStatus(done=None, current=memory.current, source="vlm", error=res.error)
        d = res.data
        done = [bool(x) for x in d["stages_done"]]
        if len(done) != K:
            return StageStatus(done=None, current=memory.current, source="vlm",
                               error=f"{len(done)} flags for {K} sub-goals")
        conf = float(d["confidence"])
        note = f"{d['event']}: {d['event_detail']}" if d.get("event", "none") != "none" and d.get("event_detail") else ""
        for k in range(memory.current, K):          # sub-goals the memory does not have as done yet
            if done[k]:
                self.streak[k] = self.streak.get(k, 0) + 1
                if self.streak[k] < self.done_streak or conf < self.min_confidence:
                    done[k] = False
            else:
                self.streak[k] = 0
        current = next((k for k, x in enumerate(done) if not x), K)
        return StageStatus(done=done, current=current, source="vlm", confidence=conf, evidence=d["evidence"],
                           hand_clear=None, note=note)
