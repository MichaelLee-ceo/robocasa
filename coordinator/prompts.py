"""The fixed texts the coordinator sends to a VLM: the system prompt, one question per decision, and the JSON schema of
each answer. context.py puts them together with what the memory holds. The system prompt, the progress question and
its schema, and the event definitions are the pilot's (coordinator/vlm/pilot.py), unchanged; the confirm, recovery and plan
questions are new (no earlier version) and use the pilot's event definitions. Every question also states its JSON
format in words, for backends that do not enforce schemas.

Schemas avoid keywords that some structured-output implementations reject (minItems, patterns); lengths and ranges
are checked by the modules."""

from .vlm import pilot

# The pilot's validated prompt (coordinator/vlm/pilot.py; Claude Opus 5.5: stage flags 0.95 correct, 0 false "done" on 60
# queries). The progress request is the pilot's Query (task, stages, frames, QUESTION) with the memory blocks added.
SYSTEM = pilot.SYSTEM
EVENTS = pilot.EVENTS
EVENT_TEXT = pilot.QUESTION[pilot.QUESTION.index("Event definitions"):]
PROGRESS_Q = pilot.QUESTION
PROGRESS_SCHEMA = pilot.SCHEMA

CONFIRM_Q = """Question: is the robot failing at the current sub-goal, so that it should be stopped and helped? Or is it
still making progress, or already done? Compare the course of this attempt (the moments in order) with what the
sub-goal needs.
%s
Answer with one JSON object and nothing else:
{"failure": true or false,
 "kind": one of %s,
 "reason": "what in the images shows it",
 "confidence": a number from 0 to 1,
 "note": "one short fact worth remembering, or empty"}""" % (EVENT_TEXT, list(EVENTS))
CONFIRM_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["failure", "kind", "reason", "confidence", "note"],
    "properties": {"failure": {"type": "boolean"}, "kind": {"type": "string", "enum": list(EVENTS)},
                   "reason": {"type": "string"}, "confidence": {"type": "number"}, "note": {"type": "string"}}}

ACTION_TEXT = {
    "retrace": "drive the robot back along the path it took, to where it was when this attempt began (what it holds "
               "is carried back and put down where it was picked up; dropped or pushed objects stay where they are), "
               "then let the policy try the sub-goal again with fresh random samples",
    "resample": "keep the robot where it is and let the policy try again with fresh random samples",
    "reinstruct": "like resample, but give the policy another sentence (write it in \"instruction\"). The policy was "
                  "trained on whole-task sentences like the task instruction and follows step-level sentences poorly",
    "backoff": "open the gripper and move the hand back a little, then let the policy try again",
    "continue": "do nothing: the alarm is wrong or the robot will manage",
    "stop": "stop helping for the rest of the episode",
}


def recover_question(options):
    opts = "\n".join(f"- {o}: {ACTION_TEXT[o]}" for o in options)
    instr = "the sentence for reinstruct, else empty" if "reinstruct" in options else "empty"
    return f"""Question: what should be done now? The policy itself cannot be changed; these are the options:
{opts}
Use what the attempts so far show: an action that already failed several times on this sub-goal is less likely to
work again, and a retrace cannot bring back objects that fell or were pushed away.
Answer with one JSON object and nothing else:
{{"action": one of {list(options)},
 "instruction": "{instr}",
 "reason": "why, from the images and the attempts so far"}}"""


def recover_schema(options):
    return {"type": "object", "additionalProperties": False, "required": ["action", "instruction", "reason"],
            "properties": {"action": {"type": "string", "enum": list(options)}, "instruction": {"type": "string"},
                           "reason": {"type": "string"}}}


PLAN_Q = """Question: write the robot's to-do list for this task, the way you would plan your own work.
Rules:
- Split the task instruction into items, using its own words. One item is one skill the robot does in one go: a
  pick-and-place ("pick the X from A and place it in B" is ONE item; picking up without placing is not an item),
  opening or closing one door or drawer, pressing one button, turning one knob, lever or faucet. A sentence with two
  such skills ("close the door and press the start button") gives two items.
  Example: "Pick the mug from the cabinet and place it under the coffee machine. Then press the start button." gives
  2 items: "Pick the mug from the cabinet and place it under the coffee machine." and "Press the start button."
- Keep the order the instruction gives. When it asks for several placements without fixing their order (e.g. one
  object of each kind in each container), write one item per placement (one object into one container: "one X and
  one Y in each of two containers" is four placements) and mark these items "any_order": true: the
  robot may do them in any order, and you will tick whichever gets done. Write each one's done condition so that it
  holds whatever order they were done in ("one tupperware holds a steak", not "the first tupperware holds a steak").
  Items in a fixed order ("Then ...") are "any_order": false.
- For each item write "done": what the camera images show when it is complete (where the object is and that the
  gripper has let go of it, the door shut, the light on).
Answer with one JSON object and nothing else:
{"items": [{"text": "...", "done": "...", "any_order": true or false}, ...]}"""
PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["text", "done"],
        "properties": {"text": {"type": "string"}, "done": {"type": "string"}, "any_order": {"type": "boolean"}}}}}}

# The item check (tracking/vlm.py): only the current to-do item (with the not-yet-done items of its any-order run),
# every chunk; its progress level is a progress dot.
PROGRESS_LEVELS = ("not_started", "reaching", "handling", "finishing", "done")
ITEM_Q = """Question: judge the CURRENT item(s), marked [>] in the to-do list, at the LAST moment ("now"). When several
items are marked [>], they may be done in any order.
- "done": for each item marked [>], does its done condition hold now? For a placement: the object rests at its goal
  and the gripper no longer holds it (the hand may still be next to it). For a door, drawer, knob, button or faucet:
  it is in its goal state. Judge from the frames only.
- "progress": how far the [>] item the robot is working on now has got:
  "not_started": the robot has not begun it (idle, or still busy with the previous item);
  "reaching": the hand is moving toward the object or fixture part this item needs;
  "handling": the gripper holds, pushes or turns it, or carries the object toward the goal;
  "finishing": the object is at or just above its goal but still held, or the fixture is nearly in its goal state;
  "done": it is done.
- "advancing": true if, over the last few moments, the robot got closer to finishing a [>] item, or one is done;
  false if it is hovering, repeating the same motion, moving away from the goal, or not moving.
- "previous_ok": true if what the items marked [x] achieved still holds (false if, e.g., a placed object fell out or
  a closed door opened again); true when there are none.
Answer with one JSON object and nothing else:
{"done": {"<number of each item marked [>]>": true or false},
 "progress": one of %s,
 "advancing": true or false,
 "previous_ok": true or false,
 "evidence": "what you see in the frames that supports the answer",
 "confidence": a number from 0 to 1}""" % (list(PROGRESS_LEVELS),)
ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["done", "progress", "advancing", "previous_ok", "evidence", "confidence"],
    "properties": {"done": {"type": ["object", "boolean"]}, "progress": {"type": "string", "enum": list(PROGRESS_LEVELS)},
                   "advancing": {"type": "boolean"}, "previous_ok": {"type": "boolean"}, "evidence": {"type": "string"},
                   "confidence": {"type": "number"}}}


TRIGGER_TEXT = {
    "failure_point": "the run starts from a moment at which this sub-goal is known to be failing",
    "timeout": "this attempt has taken {seconds:.0f} s; the sub-goal is usually done within {window_s:.0f} s",
    "undone": "sub-goal {stage1}, done earlier, no longer reads as done",
    "empty_grasp": "the gripper closed on nothing (fingers {gap_mm} mm apart)",
    "stationary": "the hand has barely moved for {seconds:.0f} s",
    "missed_grasp": "a grasp at {target} seems to have missed",
    "dropped": "the held {target} seems to have been dropped",
    "fixture_short": "a fixture was moved only part of the way",
    "wrong_element": "the robot moved a fixture part other than the target",
    "no_progress": "the robot has not got closer to finishing this item at the last {n} checks",
}
