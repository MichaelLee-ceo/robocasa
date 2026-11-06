"""Confirmers: decide whether to act on a trigger.

AcceptAll     every trigger is acted on (the oracle setting).
VLMConfirmer  reference v0 (untuned; owner: ___): context.confirm_request (the alarm, every attempt at the sub-goal
              and how it ended, the interventions, the robot's sensors, notes, the attempt's start, moments spread
              over the attempt, recent moments) -> (failure?, "kind: reason"). Triggers in `given` (the failure point
              of a point run: detection assumed solved) are accepted without a call. A failed call is acted on
              (on_error=True) or ignored. Score it against labelled triggers before trusting it (precision / recall)."""
from .. import context as C
from ..vlm import make_vlm
from .base import Confirmer


class AcceptAll(Confirmer):
    def confirm(self, memory, obs, trigger):
        return True, "accepted (no confirmer)"


class VLMConfirmer(Confirmer):
    def __init__(self, vlm, cfg=None, given=("failure_point",), on_error=True, min_confidence=0.0):
        self.vlm, self.cfg, self.given = make_vlm(vlm), cfg or C.ContextConfig(), tuple(given)
        self.on_error, self.min_confidence = on_error, min_confidence

    def confirm(self, memory, obs, trigger):
        if trigger.kind in self.given:
            return True, "given (detection assumed solved)"
        res = self.vlm.ask(C.confirm_request(memory, obs, trigger, self.cfg))
        memory.log_vlm("confirmer", obs, res)
        if not res.ok:
            return self.on_error, f"VLM error, {'acted on' if self.on_error else 'ignored'}: {res.error}"
        d = res.data
        if d.get("note"):
            memory.add_note(obs, d["note"], "confirmer", trigger.stage)
        trigger.detail["vlm_kind"] = d["kind"]
        ok = bool(d["failure"]) and float(d["confidence"]) >= self.min_confidence
        return ok, f"{d['kind']}: {d['reason']}"
