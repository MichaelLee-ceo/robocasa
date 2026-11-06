"""Undone: an earlier sub-goal reads not done at UNDONE_N tracker answers in a row, counted only when the hand is not
near an object (StageStatus.hand_clear is not False). A stage check that includes "the gripper is away from X"
reads undone while the hand is near X (e.g. pressing start next to the bowl), which sent the robot back for nothing."""
from ..schema import Trigger
from .base import Detector


class Undone(Detector):
    UNDONE_N = 3

    def __init__(self, n=UNDONE_N):
        self.n, self.low = n, 0

    def check(self, memory, obs, status):
        if status is None or status.done is None:
            return []
        earlier = [k for k in range(min(memory.current, len(status.done))) if not status.done[k]]
        if not earlier:
            self.low = 0
            return []
        if status.hand_clear is False:
            return []
        self.low += 1
        if self.low >= self.n:
            self.low = 0
            return [Trigger("undone", earlier[0], obs.chunk, dict(was=memory.current))]
        return []
