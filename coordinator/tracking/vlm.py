"""The VLM tracker (the proposal's progress tracking): a to-do list kept the way a person keeps one.

make_plan: the VLM splits the task instruction into the to-do list, one action per item in the instruction's own
    words, each with what "done" looks like and whether it may be done in any order with its neighbours (e.g. the
    placements of "one X and one Y in each container") (context.plan_request). With `plan` given (e.g.
    tracking.oracle.task_plan) that list is used instead.
update, at every check (the coordinator calls it every `track_every` chunks; 1 = every chunk):
    item check (context.item_request): the CURRENT item, and with it the not-yet-done items of its any-order run
        (memory.open_items: the policy picks the order there, and it is not told one). The VLM sees the to-do list
        (done items, the open ones with their done conditions, the current item's earlier attempts and how they
        ended, its progress dots), the robot's own sensors, and the moments of this attempt (context.item_moments).
        It answers, for each open item, whether it is done (yes / no; a placement is done when the object rests at
        its goal and the gripper no longer holds it), the progress level of the item being worked on (not_started /
        reaching / handling / finishing / done), whether it is advancing, and whether the earlier items' results
        still hold. Each answer is a progress dot in the memory (memory.dots).
    tick: an item called done at `done_streak` checks in a row (confidence >= min_confidence) is ticked: the current
        item -> the next one not done begins; another open item -> StageStatus.extra_done (the loop records it,
        memory.tick_out_of_order, and the current item begins a new attempt from that moment). One tick per check.
        (A separate hand-off check before ticking was dropped on 2026-11-08, the user's decision: it blocked 69
        correct ticks for 13 premature ones prevented, and the hand-off conditions the VLM wrote could be false by
        construction, e.g. "the other items remain on the counter" when they were in the fridge.)
    any-order items are placements: one is ticked only if the gripper has grasped and released since the attempt
        began (the robot's own gripper commands, memory.gripper_events). Each placement needs one, and the VLM tends
        to count an identical object twice (PackIdenticalLunches: "a second squash" ticked 2 chunks after the
        first); a blocked tick is noted once in the memory.
    previous_ok false: the previous item reads not done in the StageStatus (detection/undone.py decides).
A failed VLM call returns StageStatus(done=None): the coordinator keeps its current view."""
import re

from .. import context as C
from ..schema import StageStatus, SubGoal
from ..vlm import make_vlm
from .base import ProgressTracker


def done_flags(answer, items):
    """{item: bool} from the answer's "done": {"<item number>": bool} (keys like "3" or "item 3"; a bare bool
    answers for the first item)"""
    d = answer.get("done")
    if isinstance(d, bool):
        return {j: (d if j == items[0] else False) for j in items}
    out = {j: False for j in items}
    for key, v in (d or {}).items():
        m = re.search(r"\d+", str(key))
        if m and int(m.group()) - 1 in out:
            out[int(m.group()) - 1] = v is True or str(v).strip().lower() == "true"
    return out


def grasped_and_released(memory):
    """the gripper closed (or was closed) and then opened since the current attempt began (the robot trace);
    None when there is no trace"""
    a = memory.latest_start()
    if memory.trace is None or a is None or a.get("trace_index") is None:
        return None
    g = memory.trace.path[a["trace_index"]]["grip"] if a["trace_index"] < len(memory.trace.path) else None
    closed = g is not None and g > 0.5
    for _, e in memory.gripper_events(a["trace_index"]):
        if e == "closed":
            closed = True
        elif closed:
            return True
    return False


class VLMTracker(ProgressTracker):
    def __init__(self, vlm, cfg=None, plan=None, done_streak=2, min_confidence=0.0):
        self.vlm, self.cfg, self.given_plan = make_vlm(vlm), cfg or C.ContextConfig(), plan
        self.done_streak, self.min_confidence = done_streak, min_confidence
        self.streak, self.plan_call, self.noted = {}, None, set()

    def make_plan(self, instruction, obs):
        if self.given_plan is not None:
            return list(self.given_plan)
        res = self.plan_call = self.vlm.ask(C.plan_request(instruction, obs, self.cfg))
        if not res.ok:
            raise RuntimeError(f"the VLM could not make the plan: {res.error}")
        return [SubGoal(i, g["text"], g.get("done", ""), bool(g.get("any_order", False)))
                for i, g in enumerate(res.data["items"])]

    def update(self, memory, obs):
        K, k = len(memory.plan), memory.current
        if k >= K:
            return StageStatus(done=[True] * K, current=K, source="vlm")
        items = memory.open_items(k)
        res = self.vlm.ask(C.item_request(memory, obs, self.cfg))
        memory.log_vlm("tracker", obs, res)
        if not res.ok:
            return StageStatus(done=None, current=k, source="vlm", error=res.error)
        d = res.data
        conf = float(d["confidence"])
        flags = done_flags(d, items)
        some = any(flags.values())
        memory.add_dot(obs, k, "done" if some else d["progress"], d["advancing"] or some, d["previous_ok"], conf)
        attempt = (k, len(memory.stages[k].attempts))
        for j in items:
            key = (j, attempt)
            self.streak[key] = self.streak.get(key, 0) + 1 if flags[j] and conf >= self.min_confidence else 0
        tick, cycle = None, None
        for j in items:
            key = (j, attempt)
            if self.streak[key] < self.done_streak:
                continue
            if memory.plan[j].any_order:
                cycle = grasped_and_released(memory) if cycle is None else cycle
                if cycle is False:
                    if key not in self.noted:
                        self.noted.add(key)
                        memory.add_note(obs, f"item {j + 1} looked done, but the gripper has not grasped and released "
                                             f"anything since this attempt began", "tracker")
                    continue
            tick = j
            break                                   # one tick per check
        done = [memory.is_done(i) or i < k for i in range(K)]
        if k > 0 and not d["previous_ok"]:
            done[k - 1] = False
        extra, current = [], k
        if tick is not None:
            done[tick] = True
            if tick == k:
                current = next((i for i in range(k + 1, K) if not memory.is_done(i)), K)
            else:
                extra = [tick]
        return StageStatus(done=done, current=current, source="vlm", confidence=conf, evidence=d["evidence"],
                           hand_clear=None, extra_done=extra)
