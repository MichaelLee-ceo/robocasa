"""The oracle tracker: envkit's stage checks on the simulator state (the official checks, stage by stage). The
proposal's "oracle simulator-state tracker (upper bound)"; all recovery results so far use it."""
from ..paths import use_envkit
from ..schema import StageStatus, SubGoal
from .base import ProgressTracker

use_envkit()
import robocasa.utils.object_utils as OU  # noqa: E402
import stages  # noqa: E402


def task_plan(ep):
    """the task's stage sentences (envkit stages.instructions, computed once at the start): a given checklist, for
    trackers that do not build their own"""
    return [SubGoal(index=i, text=t) for i, t in enumerate(stages.instructions(ep))]


def hand_clear(e):
    """(simulator) the hand touches no object and is farther than 0.25 m from every object (the distance the stage
    checks use for "away"): a "not done" seen while the hand is near may only mean "the hand is too close"."""
    g = e.robots[0].gripper["right"]
    return all(OU.gripper_obj_far(e, o) and not e.check_contact(g, e.objects[o]) for o in e.objects)


class OracleTracker(ProgressTracker):
    uses_sim = True

    def __init__(self, ep):
        self.ep = ep

    def make_plan(self, instruction, obs):
        return task_plan(self.ep)

    def update(self, memory, obs):
        tr = stages.track(self.ep)
        return StageStatus(done=[bool(d) for d in tr["done"]], current=int(tr["stage"]), source="oracle",
                           hand_clear=hand_clear(self.ep.kenv))
