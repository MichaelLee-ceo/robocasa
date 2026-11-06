"""VLMRecovery: the VLM picks the intervention. Reference v0 (untuned; owner: ___).

context.recovery_request gives it the alarm (and the confirmer's reason), every attempt at the sub-goal and how it
ended, the interventions already tried, the robot's sensors, notes, the sub-goal's first start, the attempt's start and
recent moments, and the options in `options` with what each does. Over the episode budget -> stop. A failed call ->
the `fallback` policy (default RetraceRetry). reinstruct without a sentence -> the sub-goal's sentence."""
from .. import context as C
from ..schema import Intervention
from ..vlm import make_vlm
from .base import EXECUTABLE, RecoveryPolicy, used
from .policies import RetraceRetry


class VLMRecovery(RecoveryPolicy):
    def __init__(self, vlm, cfg=None, options=("retrace", "resample", "reinstruct", "continue"), budget=None,
                 fallback=None):
        bad = [o for o in options if o not in EXECUTABLE]
        if bad:
            raise ValueError(f"not executable yet: {bad}")
        self.vlm, self.cfg, self.options = make_vlm(vlm), cfg or C.ContextConfig(), tuple(options)
        self.budget, self.fallback = budget, fallback or RetraceRetry()

    def decide(self, memory, obs, trigger):
        if self.budget is not None and len(used(memory)) >= self.budget:
            return Intervention("stop", trigger.stage, reason="budget used up")
        res = self.vlm.ask(C.recovery_request(memory, obs, trigger, self.options, self.cfg))
        memory.log_vlm("recovery", obs, res)
        if not res.ok:
            iv = self.fallback.decide(memory, obs, trigger)
            iv.reason = f"VLM error, fallback: {res.error}"
            return iv
        d = res.data
        instr = (d.get("instruction") or memory.plan[trigger.stage].text) if d["action"] == "reinstruct" else None
        return Intervention(d["action"], trigger.stage, instruction=instr, reason=d["reason"])
