"""StageTimeout: GR00T's own time on the current attempt exceeds the sub-goal's window, and again at every further
window (so an alarm dismissed by a confirmer comes back if the robot stays stuck). Windows: the 90th percentile of the
sub-goal's completion time in phase 0 (runs/20261105_recovery/phase0/plain.jsonl; what all results so far use), or
given per (task, stage) in chunks. A stage without phase 0 data gets `default`."""
import json, math

import numpy as np

from ..paths import PHASE0_PLAIN
from ..schema import STEPS_PER_CHUNK, Trigger
from .base import Detector


def phase0_windows(q=90, path=PHASE0_PLAIN):
    """{(task, stage): chunks}: the q-th percentile of each stage's completion time in phase 0"""
    dur = {}
    for line in open(path):
        r = json.loads(line)
        fd = {int(k): v for k, v in r.get("first_done", {}).items()}
        for k in range(r.get("n_stages", 0)):
            if k in fd and (k == 0 or k - 1 in fd):
                dur.setdefault((r["task"], k), []).append((fd[k] - (fd[k - 1] if k else 0)) / STEPS_PER_CHUNK)
    return {k: math.ceil(np.percentile(v, q)) for k, v in dur.items()}


class StageTimeout(Detector):
    def __init__(self, task, windows=None, default=40):
        self.task, self.windows, self.default = task, windows if windows is not None else phase0_windows(), default
        self.fired = {}  # (stage, attempt number) -> how many windows already fired

    def window(self, k):
        return self.windows.get((self.task, k), self.default)

    def check(self, memory, obs, status):
        k = memory.current
        if k >= len(memory.plan):
            return []
        key = (k, len(memory.stages[k].attempts))
        n = memory.time_on_stage(obs) // (self.window(k) * STEPS_PER_CHUNK)
        if n > self.fired.get(key, 0):
            self.fired[key] = n
            return [Trigger("timeout", k, obs.chunk, dict(window_chunks=self.window(k), windows=int(n)))]
        return []
