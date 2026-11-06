"""Memory -> VLM request, one builder per decision. This is where the memory is put to use: each request carries the
parts of the memory that decision needs, labelled, in a fixed order.

Every request is built on the pilot's Query (coordinator/vlm/pilot.py): its intro (task instruction, the sub-goals in order,
how many moments and cameras), its frames (one line "t = .. s (chunk ..), <camera>:" before each image, oldest first),
and, for progress, its validated question and schema. The memory blocks go between the intro and the frames, and a
line naming each moment ("--- now ---", "--- when this attempt began ---") goes before its frames.

| block (function)        | content                                                                | progress | confirm | recovery |
|-------------------------|------------------------------------------------------------------------|----------|---------|----------|
| (pilot intro)           | task sentence; the sub-goals in order; moments and cameras             | yes      | yes     | yes      |
| state_block             | each sub-goal's state from the memory (done at, CURRENT: attempt n,    | yes      | yes     | yes      |
|                         | time in it vs. its usual time, not started)                            |          |         |          |
| last_check_block        | the tracker's previous verdict and evidence on the current sub-goal    | yes      |         |          |
| robot_block             | the robot's own sensors: gripper, finger gap, likely holding, motion   | yes      | yes     | yes      |
|                         | recently and since the attempt began, gripper changes in the attempt   |          |         |          |
| notes_block             | the notes kept so far                                                  | yes      | yes     | yes      |
| attempts_block          | every attempt at the current sub-goal and how it ended; interventions  |          | yes     | yes      |
| trigger_block           | the alarm in words                                                     |          | yes     | yes      |
| moments (images)        | keyframes: the sub-goal's first start, the attempt's start, the        | key +    | attempt | first    |
|                         | previous sub-goal's completion; spread over the attempt (history);     | recent   | start + | start +  |
|                         | recent chunk ends; the last one labelled "now"                         |          | history | attempt  |
|                         |                                                                        |          | + recent| + recent |
Each moment is a text line (label, time, chunk) followed by one image per camera, each after its camera name.

ContextConfig sets how much goes in (cameras, number of recent moments, history, which blocks)."""
from dataclasses import dataclass, field

from . import prompts as P
from .schema import CAMERAS, CONTROL_HZ, STEPS_PER_CHUNK
from .vlm.base import VLMRequest
from .vlm import pilot


@dataclass
class ContextConfig:
    cameras: tuple = tuple(CAMERAS)          # which views; fewer = cheaper
    n_recent: int = 3                        # recent chunk-end moments (the last is "now")
    recent_stride: int = 1                   # chunks between them
    progress_state: bool = True              # the state block in the progress request (False: the pilot's query)
    keyframes: bool = True
    history: int = 3                         # moments spread over the current attempt (confirm, recovery)
    robot_state: bool = True
    notes: bool = True
    last_check: bool = True
    max_tokens: int = None                   # None: the backend's default


def _s(t):
    return f"{t / CONTROL_HZ:.1f} s"


# ---------------------------------------------------------------------------------------------------- text blocks
def state_block(m, obs):
    lines = ["Progress so far (the coordinator's memory):"]
    for k, st in enumerate(m.stages):
        a = st.open_attempt()
        if k == m.current and a is not None:
            line = f"  {k + 1}. CURRENT: attempt {len(st.attempts)}, {_s(obs.t - a['t'])} into it"
            if k in m.expected:
                line += f" (usually done within {m.expected[k] * STEPS_PER_CHUNK / CONTROL_HZ:.0f} s)"
        elif st.done_times and not (st.undone_times and st.undone_times[-1]["t"] > st.done_times[-1]["t"]):
            line = f"  {k + 1}. done at {_s(st.done_times[-1]['t'])}"
        elif st.undone_times:
            line = f"  {k + 1}. was done, then undone at {_s(st.undone_times[-1]['t'])}"
        else:
            line = f"  {k + 1}. not started"
        if st.subgoal.check:
            line += f" (complete when: {st.subgoal.check})"
        lines.append(line)
    if m.current >= len(m.stages):
        lines.append("All sub-goals have been recorded as done.")
    return "\n".join(lines)


def last_check_block(m):
    k = min(m.current, len(m.stages) - 1)
    ls = m.stages[k].last_status
    if not ls or not ls.get("evidence"):
        return ""
    return (f"Your previous check of sub-goal {k + 1} (at {_s(ls['t'])}): {'done' if ls['done'] else 'not done'}; "
            f"you saw: {ls['evidence']}")


def robot_block(m, obs):
    r = m.robot_state(obs)
    s = f"Robot sensors now: gripper {r['gripper']}"
    if r.get("finger_gap_mm") is not None:
        s += f", fingers {r['finger_gap_mm']:.0f} mm apart"
    if "holding" in r:
        s += f", holding something: {r['holding']}"
    s += "."
    if "hand_moved_cm_recent" in r:
        s += (f" Over the last {r['recent_seconds']:.1f} s the hand moved {r['hand_moved_cm_recent']:.0f} cm "
              f"(relative to the base) and the base {r['base_moved_cm_recent']:.0f} cm.")
    if "hand_from_attempt_start_cm" in r:
        s += (f" Since this attempt began: hand {r['hand_from_attempt_start_cm']:.0f} cm, base "
              f"{r['base_from_attempt_start_cm']:.0f} cm from where they were")
        ch = r.get("gripper_changes_this_attempt")
        s += f"; gripper {', then '.join(ch)}." if ch else "; no gripper change."
    return s


def notes_block(m, last=8):
    if not m.notes:
        return ""
    return "Notes kept so far:\n" + "\n".join(f"  - at {_s(n['t'])} (sub-goal {n['stage'] + 1}): {n['text']}"
                                              for n in m.notes[-last:])


def attempts_block(m, k):
    lines = [f"Attempts at sub-goal {k + 1}:"]
    for a in m.attempt_summary(k):
        why = {"begin": "first time", "replay": "first time"}.get(a["why"], f"after {a['why']}")
        if a["how"] is None:
            lines.append(f"  {a['n']}. began at {a['began_s']:.1f} s ({why}); in progress")
        else:
            lines.append(f"  {a['n']}. began at {a['began_s']:.1f} s ({why}); ended at {a['ended_s']:.1f} s after "
                         f"{a['seconds']:.1f} s: {a['how']}")
    ivs = [iv for iv in m.interventions if iv["stage"] == k]
    if ivs:
        lines.append(f"Interventions on sub-goal {k + 1}:")
        for iv in ivs:
            res = iv.get("result") or {}
            extra = "" if "arrived" not in res else (" (arrived back)" if res["arrived"] else " (did not get all the way back)")
            lines.append(f"  - at {_s(iv['t'])}: {iv['kind']}{extra}" + (f"; reason: {iv['reason']}" if iv.get("reason") else ""))
    return "\n".join(lines)


def trigger_block(m, obs, trig):
    d = dict(trig.detail)
    d.setdefault("seconds", m.time_on_stage(obs) / CONTROL_HZ)
    d.setdefault("window_s", m.expected.get(trig.stage, 0) * STEPS_PER_CHUNK / CONTROL_HZ)
    d.setdefault("stage1", trig.stage + 1)
    d.setdefault("target", "the object")
    d.setdefault("gap_mm", "?")
    try:
        what = P.TRIGGER_TEXT.get(trig.kind, trig.kind).format(**d)
    except (KeyError, ValueError):
        what = trig.kind
    s = f"Alarm at {_s(obs.t)} on sub-goal {trig.stage + 1}: {what}."
    if trig.reason and trig.confirmed:
        s += f" It was confirmed as a failure: {trig.reason}"
    return s


# ---------------------------------------------------------------------------------------------------- images
def _moments(m, obs, cfg, keys=(), history=0, n_recent=None):
    """[(label, Obs)], oldest first, one per chunk; the last is (now, obs)"""
    out = []
    for key, label in keys:
        o = m.keyframes.get(key)
        if o is not None:
            out.append((label, o))
    if history:
        h = [o for o in m.stages[min(m.current, len(m.stages) - 1)].history]
        step = max(1, len(h) // history)
        out += [("during this attempt", o) for o in h[::step][-history:]]
    rec = m.recent(n_recent if n_recent is not None else cfg.n_recent, cfg.recent_stride)
    out += [("recent", o) for o in rec] + [("now", obs)]
    by = {}
    for label, o in out:                       # one moment per chunk; its labels merged
        o0, labels = by.setdefault(o.chunk, (o, []))
        if label not in labels:
            labels.append(label)
    res = []
    for c in sorted(by):
        o, labels = by[c]
        named = [lb for lb in labels if lb not in ("recent", "during this attempt", "now")]
        if c == obs.chunk:
            o, named = obs, ["now"] + named
        res.append(("; ".join(named) or labels[0], o))
    return res


def pilot_parts(m, moments, cameras, qid):
    """(the pilot Query's intro, its frame parts with a line naming each moment before the moment's frames)"""
    frames = [(o.t / CONTROL_HZ, o.chunk, c, o.images[c]) for _, o in moments for c in cameras if c in o.images]
    parts = pilot.Query(qid, m.instruction, [g.text for g in m.plan], frames).parts()
    intro, body, out, i = parts[0], parts[1:-1], [], 0
    for label, o in moments:
        n = 2 * sum(1 for c in cameras if c in o.images)
        out += [("text", f"--- {label} ---")] + body[i:i + n]
        i += n
    return intro, out


def _join(*blocks):
    return "\n\n".join(b for b in blocks if b)


# ---------------------------------------------------------------------------------------------------- requests
def progress_request(m, obs, cfg=ContextConfig()):
    k = min(m.current, len(m.stages) - 1)
    keys = [(f"s{k}_start", f"when sub-goal {k + 1} first began"), (f"s{k - 1}_done", f"when sub-goal {k} was done")] \
        if cfg.keyframes else []
    intro, frames = pilot_parts(m, _moments(m, obs, cfg, keys), cfg.cameras, f"progress_{obs.chunk}")
    head = _join(state_block(m, obs) if cfg.progress_state else "", last_check_block(m) if cfg.last_check else "",
                 robot_block(m, obs) if cfg.robot_state else "", notes_block(m) if cfg.notes else "")
    parts = [intro] + ([("text", head)] if head else []) + frames + [("text", P.PROGRESS_Q)]
    return VLMRequest(parts=parts, system=P.SYSTEM, schema=P.PROGRESS_SCHEMA, max_tokens=cfg.max_tokens, tag="progress")


def confirm_request(m, obs, trig, cfg=ContextConfig()):
    k = trig.stage
    keys = [(f"s{k}_attempt", "when this attempt began")] if cfg.keyframes else []
    intro, frames = pilot_parts(m, _moments(m, obs, cfg, keys, history=cfg.history), cfg.cameras, f"confirm_{obs.chunk}")
    head = _join(state_block(m, obs), trigger_block(m, obs, trig), attempts_block(m, k),
                 robot_block(m, obs) if cfg.robot_state else "", notes_block(m) if cfg.notes else "")
    parts = [intro, ("text", head)] + frames + [("text", P.CONFIRM_Q)]
    return VLMRequest(parts=parts, system=P.SYSTEM, schema=P.CONFIRM_SCHEMA, max_tokens=cfg.max_tokens, tag="confirm")


def recovery_request(m, obs, trig, options, cfg=ContextConfig()):
    k = trig.stage
    keys = [(f"s{k}_start", f"when sub-goal {k + 1} first began"), (f"s{k}_attempt", "when this attempt began")] \
        if cfg.keyframes else []
    moments = _moments(m, obs, cfg, keys, history=min(cfg.history, 2), n_recent=2)
    intro, frames = pilot_parts(m, moments, cfg.cameras, f"recovery_{obs.chunk}")
    head = _join(state_block(m, obs), trigger_block(m, obs, trig), attempts_block(m, k),
                 robot_block(m, obs) if cfg.robot_state else "", notes_block(m) if cfg.notes else "")
    parts = [intro, ("text", head)] + frames + [("text", P.recover_question(options))]
    return VLMRequest(parts=parts, system=P.SYSTEM, schema=P.recover_schema(options), max_tokens=cfg.max_tokens,
                      tag="recovery")


def plan_request(instruction, obs, cfg=ContextConfig()):
    frames = [(obs.t / CONTROL_HZ, obs.chunk, c, obs.images[c]) for c in cfg.cameras if c in obs.images]
    body = pilot.Query("plan", instruction, [], frames).parts()[1:-1]
    parts = [("text", f"Task instruction: \"{instruction}\"\nThe robot's camera views at the start:")] + body + \
        [("text", P.PLAN_Q)]
    return VLMRequest(parts=parts, system=P.SYSTEM, schema=P.PLAN_SCHEMA, max_tokens=cfg.max_tokens, tag="plan")
