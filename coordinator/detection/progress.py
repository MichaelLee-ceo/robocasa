"""A trigger from the VLM tracker's to-do list (tracking/vlm.py), read from the memory:
ProgressStall   the progress dots of the current attempt: not advancing at the last `n` checks, at least `min_chunks`
                into the attempt ("no_progress"). It fires again only after `n` more dots.
It needs a tracker that writes memory.dots (the VLM tracker); with another tracker it never fires."""
from ..schema import Trigger
from .base import Detector


class ProgressStall(Detector):
    def __init__(self, n=6, min_chunks=4):
        self.n, self.min_chunks, self.used = n, min_chunks, {}

    def check(self, memory, obs, status):
        k = memory.current
        if status is None or k >= len(memory.plan):
            return []
        dots = memory.attempt_dots(k)
        key = (k, len(memory.stages[k].attempts))
        fresh = dots[self.used.get(key, 0):]
        a = memory.latest_start(k)
        if len(fresh) < self.n or a is None or obs.chunk - a["chunk"] < self.min_chunks:
            return []
        if any(d["advancing"] or d["progress"] == "done" for d in fresh[-self.n:]):
            return []
        self.used[key] = len(dots)
        return [Trigger("no_progress", k, obs.chunk, dict(n=self.n, progress=fresh[-1]["progress"]))]
