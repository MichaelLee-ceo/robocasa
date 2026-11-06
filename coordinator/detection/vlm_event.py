"""VLMEvent: the VLM's failure report as an early trigger (2026-11-06; reference version, thresholds from the offline
evaluation in runs/20261106_vlm_offline).

Every `every` chunks while a sub-goal is open, the VLM is asked the progress question (the pilot's: stage flags and an
event: missed_grasp, dropped, fixture_short, wrong_element, no_progress, other, or none) with the memory the context
config allows. `n` answers in a row reporting an event on the current sub-goal, at least `min_chunks` of GR00T time
into the attempt, make a trigger whose kind is the last event. The answer's stage flags are not used here: the tracker
tracks. The StageTimeout stays as the fallback, so this can only make detection earlier (and add false alarms)."""
from .. import context as C
from ..schema import STEPS_PER_CHUNK, Trigger
from ..vlm import make_vlm
from .base import Detector


class VLMEvent(Detector):
    def __init__(self, vlm, cfg=None, every=4, n=2, min_chunks=4, events=None):
        self.vlm, self.cfg = make_vlm(vlm), cfg or C.ContextConfig()
        self.every, self.n, self.min_chunks, self.events = every, n, min_chunks, set(events) if events else None
        self.streak, self.calls = {}, 0

    def check(self, memory, obs, status):
        k = memory.current
        if k >= len(memory.plan):
            return []
        key = (k, len(memory.stages[k].attempts))
        if memory.time_on_stage(obs) < self.min_chunks * STEPS_PER_CHUNK:
            self.streak[key] = 0
            return []
        self.calls += 1
        if (self.calls - 1) % self.every:
            return []
        res = self.vlm.ask(C.progress_request(memory, obs, self.cfg))
        memory.log_vlm("vlm_event", obs, res)
        if not res.ok:
            return []
        ev = res.data.get("event", "none")
        if ev != "none" and (self.events is None or ev in self.events):
            self.streak[key] = self.streak.get(key, 0) + 1
        else:
            self.streak[key] = 0
        if self.streak[key] >= self.n:
            self.streak[key] = 0
            return [Trigger(ev, k, obs.chunk, dict(detail=res.data.get("event_detail", ""), source="vlm",
                                                   evidence=res.data.get("evidence", "")))]
        return []
