"""envkit's simulator events (missed_grasp, dropped, fixture_short, wrong_element) as triggers: a labelled upper bound
for the proprio triggers, and their ground truth."""
from ..paths import use_envkit
from ..schema import Trigger
from .base import Detector


class OracleEvents(Detector):
    uses_sim = True

    def __init__(self, ep):
        use_envkit()
        import events
        self.ep, self.ev = ep, events.Events(ep)

    def check(self, memory, obs, status):
        return [Trigger(e["type"], int(e["stage"]), obs.chunk, {k: v for k, v in e.items() if k not in ("type", "stage")})
                for e in self.ev.update(self.ep)]
