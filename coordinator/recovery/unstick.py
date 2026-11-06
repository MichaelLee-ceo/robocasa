"""Unstick: when a retrace cannot follow the recorded path (the arm blocked, e.g. wedged in an open fridge), GR00T gets
one sentence for a few chunks to free the arm, then the retrace resumes along the recorded path (loop.py). The user's
design (2026-11-07): the deterministic retrace by default; when it fails, anything that might still work, and the VLM
chooses the sentence.

VLMUnstick    the VLM sees the views now (the arm blocked) and when the sub-goal began (where the robot is going back
              to), and writes one sentence in GR00T's own wording (gr00t_wording: only bank words and this task's
              words, level L1). An invalid answer is asked again once with the reason; then the task sentence is used.
FixedUnstick  a given sentence (None = the task sentence).
choose(memory, obs, stage) -> (instruction or None for the task sentence, reason)

Measured first by hand at PackIdenticalLunches s4 (runs/20261106_coordinator/README.md): "Close the fridge door." freed
the arm in 3 of 4 GR00T seeds, the task sentence in 1 of 4, backing the base up in 0 of 1."""
import re

from .. import gr00t_wording as W
from ..schema import CAMERAS, CONTROL_HZ
from ..vlm import make_vlm
from ..vlm.base import VLMRequest

SYSTEM = ("You help a mobile kitchen robot recover. You see its three cameras (two shoulder views and a wrist camera) "
          "and answer with one JSON object.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["instruction", "reason"],
          "properties": {"instruction": {"type": "string"}, "reason": {"type": "string"}}}
CAMERA_NAMES = {"video.robot0_agentview_left": "left shoulder view", "video.robot0_agentview_right": "right shoulder view",
                "video.robot0_eye_in_hand": "wrist camera"}


def scene_words(memory):
    text = " ".join([memory.instruction] + [g.text for g in memory.plan])
    return set(W.words(text))


def question(memory, stage, scene, free=False):
    bank = "\n".join(f"  - {t}" for t in W.BANK.values())
    words = ("You may use any words." if free else
             f"Use only words from the sentences above and these words of this task: {', '.join(sorted(scene))}.")
    return (f"Task: \"{memory.instruction}\"\nThe robot was driving its arm back along the path it came by, to where it "
            f"was when this step began: \"{memory.plan[stage].text}\" (the views marked \"goal\"), so that it can try "
            f"that step again. The arm is now blocked and cannot follow the path (the views marked \"now\").\n"
            f"The robot is driven by a policy that takes only one sentence. It was trained on sentences like these "
            f"({{}} is a place for an object or fixture word):\n{bank}\n"
            f"Write one sentence for the policy that should make it move the arm free of what blocks it. It does not "
            f"have to be about this step: any step (the next one, or one with another object or fixture) that would "
            f"move the arm out of where it is stuck will do. Once the arm is free, the robot goes back and tries this "
            f"step again. {words}\n"
            "Answer with one JSON object and nothing else:\n"
            "{\"instruction\": \"the sentence\", \"reason\": \"what blocks the arm, and why this sentence should free it\"}")


class FixedUnstick:
    def __init__(self, instruction=None):
        self.instruction = instruction

    def choose(self, memory, obs, stage):
        return self.instruction, "fixed"


class VLMUnstick:
    def __init__(self, vlm, cameras=tuple(CAMERAS), level="L1"):
        """level: the wording rule (gr00t_wording: L0, L1, L2), or None for free wording"""
        self.vlm, self.cameras, self.level = make_vlm(vlm), cameras, level

    def _parts(self, memory, obs, stage, extra=""):
        parts = []
        for label, o in (("goal", memory.keyframes.get(f"s{stage}_start")), ("now", obs)):
            if o is None:
                continue
            parts.append(("text", f"--- {label} (t = {o.t / CONTROL_HZ:.1f} s) ---"))
            for c in self.cameras:
                if c in o.images:
                    parts += [("text", f"{CAMERA_NAMES.get(c, c)}:"), ("image", o.images[c])]
        q = question(memory, stage, scene_words(memory), free=self.level is None) + (f"\n{extra}" if extra else "")
        return parts + [("text", q)]

    def choose(self, memory, obs, stage):
        scene, extra = scene_words(memory), ""
        for _ in range(2):
            res = self.vlm.ask(VLMRequest(parts=self._parts(memory, obs, stage, extra), system=SYSTEM, schema=SCHEMA,
                                          max_tokens=600, tag="unstick"))
            memory.log_vlm("unstick", obs, res)
            if not res.ok:
                return None, f"VLM error: {res.error}"
            text = re.sub(r"\s+", " ", str(res.data.get("instruction", ""))).strip()
            ok, why = (bool(text), "empty") if self.level is None else W.validate(text, self.level, scene)
            if ok:
                return text, res.data.get("reason", "")
            extra = f"Your sentence \"{text}\" is not allowed ({why}). Write another one."
        return None, f"no valid sentence ({why}); the task sentence"
