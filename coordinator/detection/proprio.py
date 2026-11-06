"""The proposal's cheap proprio triggers. TODO (owner: ___). Proprio only (obs.proprio, obs.gripper_cmd, memory.recent,
memory.trace): no simulator. Score them against detection.oracle_events.OracleEvents on recorded runs (precision /
recall, how many chunks earlier than the timeout), then add them to the configs (configs.py: proprio_triggers)."""
from ..schema import Trigger  # noqa: F401
from .base import Detector

EMPTY_GAP = 0.004   # m: the proposal's "closed gripper with a finger gap under 4 mm"


class EmptyGrasp(Detector):
    """gripper commanded closed (obs.gripper_cmd > 0.5) and obs.finger_gap() < EMPTY_GAP for a chunk or two, once
    per grasp -> [Trigger("empty_grasp", memory.current, obs.chunk, dict(gap_mm=...))]"""

    def check(self, memory, obs, status):
        raise NotImplementedError("EmptyGrasp: see the class doc")


class Stationary(Detector):
    """the hand (obs.hand(), relative to the base) and the base (obs.base()) moved less than `min_motion` m over the
    last `h_chunks` chunk ends (memory.recent) while the sub-goal is not done ->
    [Trigger("stationary", memory.current, obs.chunk, dict(seconds=...))]"""

    def __init__(self, h_chunks=6, min_motion=0.01):
        self.h, self.min_motion = h_chunks, min_motion

    def check(self, memory, obs, status):
        raise NotImplementedError("Stationary: see the class doc")
