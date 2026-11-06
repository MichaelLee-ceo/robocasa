from ..schema import Intervention
from .base import RecoveryPolicy, used


class RetraceRetry(RecoveryPolicy):
    """every confirmed trigger on sub-goal k -> retrace to where k's attempt began and retry with the task sentence and
    fresh noise; unlimited unless `budget` (interventions per episode) is given. The method behind the current result
    (runs/20261105_recovery, retrace v2: fixed 12 % -> 37 %, success 4 % -> 22 % against same-scene reruns)."""

    def __init__(self, budget=None):
        self.budget = budget

    def decide(self, memory, obs, trigger):
        if self.budget is not None and len(used(memory)) >= self.budget:
            return Intervention("stop", trigger.stage, reason="budget used up")
        return Intervention("retrace", trigger.stage)


class Ladder(RecoveryPolicy):
    """the proposal's order: each step at most `per_step` times per sub-goal, at most `budget` per episode, then stop"""

    def __init__(self, steps=("resample", "backoff", "reinstruct"), per_step=2, budget=6):
        self.steps, self.per_step, self.budget = list(steps), per_step, budget

    def decide(self, memory, obs, trigger):
        if len(used(memory)) >= self.budget:
            return Intervention("stop", trigger.stage, reason="budget used up")
        done = [iv["kind"] for iv in used(memory, trigger.stage)]
        for kind in self.steps:
            if done.count(kind) < self.per_step:
                text = memory.plan[trigger.stage].text if kind == "reinstruct" else None
                return Intervention(kind, trigger.stage, instruction=text)
        return Intervention("stop", trigger.stage, reason="ladder used up")
