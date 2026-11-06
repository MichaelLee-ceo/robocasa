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


PLAN_Q = """Question: split the task instruction into the sub-goals the robot must complete, in order. For each, give a
short instruction-like sentence and the condition that shows in the camera images when it is complete.
Answer with one JSON object and nothing else:
{"subgoals": [{"text": "...", "check": "..."}, ...]}"""
PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["subgoals"],
    "properties": {"subgoals": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["text", "check"],
        "properties": {"text": {"type": "string"}, "check": {"type": "string"}}}}}}

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
}
