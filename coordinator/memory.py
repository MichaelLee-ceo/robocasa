"""The coordinator's memory M_t (proposal §3): everything the coordinator knows about the episode so far, kept so
that a VLM can make its decisions from it (context.py turns it into requests).

It is used by the coordinator only and never goes into GR00T's input, so the executor stays in its training
distribution. Modules read it; the coordinator loop writes it (modules write only their notes and VLM call logs).

What it keeps, and which decision it serves (T = tracking "where am I?", C = confirming "is this a failure?",
R = recovery "what now?", X = the retrace mechanism):
| field / method            | content                                                                       | for     |
|---------------------------|-------------------------------------------------------------------------------|---------|
| instruction, plan         | the task sentence; per sub-goal a sentence and a completion condition         | T C R   |
| stages[k].attempts        | every attempt at sub-goal k: when it began, why (first time, after a retrace, | C R X   |
|                           | ...), the trace entry to go back to, how it ended (done, timeout, undone, ...)|         |
| stages[k].done_times      | when it was done (more than once if it was undone and redone)                 | T       |
| stages[k].last_status     | the tracker's latest verdict and evidence for this sub-goal                   | T       |
| expected[k]               | how long the sub-goal usually takes (chunks; the timeout window)              | C R     |
| keyframes                 | the scene when each sub-goal first began (s<k>_start), when its latest        | T C     |
|                           | attempt began (s<k>_attempt), when it was done (s<k>_done), at triggers       |         |
| stages[k].history         | every chunk end of the current attempt (reset when an attempt begins): the    | T C R   |
|                           | course of the attempt, not only its end (the tracker samples it uniformly)    |         |
| frames                    | the last 24 chunk-end observations (3 views + proprio)                        | T C R   |
| trace                     | the robot's own state after every env step (7 arm joints, base pose, hand     | X; C R  |
|                           | pose, gripper command): the retrace's path; gripper events and motion         | (robot) |
| notes                     | short observations the VLM asked to keep ("the vegetable fell behind the      | T C R   |
|                           | sink"), with time and sub-goal                                                |         |
| dots                      | the progress dots: the tracker's view of the current sub-goal at each check   | T C R   |
|                           | (not_started / reaching / handling / finishing / done, advancing or not)      |         |
| triggers, interventions   | what fired, what the confirmer said and why, what was done and its result     | C R     |
| status_log, vlm_calls     | every tracker answer; every VLM call (module, latency, tokens, cost)          | analysis|
| free_steps                | env steps spent retracing (not GR00T's time)                                  | clock   |
| gr00t_instruction, stream | the sentence and the noise stream GR00T gets now                              | executor|
Derived views for the VLM: robot_state(obs) (gripper, holding?, recent motion, since the attempt began),
gripper_events(since), attempt_summary(k), time_on_stage(obs)."""
import collections

import numpy as np

from .schema import CONTROL_HZ, STEPS_PER_CHUNK, Obs, to_json

KEEP_FRAMES = 24
EMPTY_GAP = 0.004          # m: closed with the fingers nearer than this = closed on nothing (the proposal's trigger)


class StageRecord:
    def __init__(self, subgoal):
        self.subgoal = subgoal
        self.attempts = []          # dict(chunk, t, trace_index, why, end=None | dict(chunk, t, how))
        self.done_times = []        # dict(chunk, t)
        self.undone_times = []      # dict(chunk, t)
        self.interventions = 0
        self.history = []           # every chunk-end Obs of the current attempt (reset when an attempt begins or the sub-goal is done)
        self.last_status = None     # dict(chunk, t, done, evidence, source)

    @property
    def done_at(self):
        return self.done_times[0] if self.done_times else None

    def open_attempt(self):
        return self.attempts[-1] if self.attempts and self.attempts[-1]["end"] is None else None

    def json(self):
        return dict(index=self.subgoal.index, text=self.subgoal.text, check=self.subgoal.check, attempts=self.attempts,
                    done_times=self.done_times, undone_times=self.undone_times, interventions=self.interventions,
                    last_status=self.last_status)


class Memory:
    def __init__(self, instruction, plan, trace=None, expected=None):
        self.instruction, self.plan, self.trace = instruction, list(plan), trace
        self.stages = [StageRecord(g) for g in self.plan]
        self.expected = dict(expected or {})      # sub-goal -> chunks
        self.current = 0
        self.frames = collections.deque(maxlen=KEEP_FRAMES)
        self.keyframes = {}
        self.notes, self.status_log, self.triggers, self.interventions, self.vlm_calls = [], [], [], [], []
        self.free_steps = 0
        self.gr00t_instruction = None
        self.stream = "start"          # GR00T's noise stream, named as in runs/20261105_recovery/loop.py
        self.stage_interventions = 0   # interventions since the current sub-goal became current (loop.py `attempts`)
        self.entry = {}                # sub-goal -> trace entry when it was entered by the replay or by moving on
                                       # (loop.py `starts`)
        self.back_to = None            # where the next retrace of the current sub-goal goes back to (loop.py `back_to`):
                                       # its entry; after a retrace that arrived, the arrival point
        self.moved_on = []             # dict(frm, to, chunk, t): every "moved on" (loop.py's fixed rule reads it)
        self.dots = []                 # dict(chunk, t, stage, attempt, progress, advancing, previous_ok, confidence)

    # ------------------------------------------------------------------ written by the coordinator loop
    def observe(self, obs: Obs):
        self.frames.append(obs)
        if self.current < len(self.stages):
            h = self.stages[self.current].history
            if not h or obs.chunk > h[-1].chunk:
                h.append(obs)

    def trace_index(self):
        return len(self.trace.path) - 1 if self.trace is not None else None

    def begin_stage(self, k, obs: Obs, why="begin", trace_index=None):
        """a (new) attempt at sub-goal k: its time counts from here; trace_index is where a retrace of this
        sub-goal goes back to (default: the robot's state now)"""
        self.end_attempt(k, obs, why)
        self.current = k
        ti = self.trace_index() if trace_index is None else trace_index
        if why in ("begin", "replay"):                 # entered: loop.py remember(k); back_to = starts[k]
            self.entry[k] = self.back_to = ti
        self.stages[k].attempts.append(dict(chunk=obs.chunk, t=obs.t, trace_index=ti, why=why, end=None))
        self.stages[k].history = [obs]                 # the course of this attempt only
        self.keyframes.setdefault(f"s{k}_start", obs)
        self.keyframes[f"s{k}_attempt"] = obs

    def end_attempt(self, k, obs: Obs, how):
        a = self.stages[k].open_attempt()
        if a is not None:
            a["end"] = dict(chunk=obs.chunk, t=obs.t, how=how)

    def move_on(self, k, obs: Obs):
        """the tracker found every sub-goal before k done: close them, begin k (loop.py: "moved on")"""
        self.moved_on.append(dict(frm=self.current, to=k, chunk=obs.chunk, t=obs.t))
        for j in range(self.current + 1, min(k, len(self.stages))):   # loop.py remembers every stage it passes
            self.entry[j] = self.trace_index()
        for j in range(self.current, min(k, len(self.stages))):
            self.complete_stage(j, obs)
        self.stream, self.stage_interventions, self.gr00t_instruction = f"s{k}", 0, None
        if k < len(self.stages):
            self.begin_stage(k, obs)
        else:
            self.current = len(self.stages)

    def tick_out_of_order(self, j, obs: Obs):
        """sub-goal j of the current any-order run, done before the current one: recorded done; the current sub-goal
        begins a new attempt here, so that a retry comes back to this moment and does not undo sub-goal j"""
        self.moved_on.append(dict(frm=j, to=self.current, chunk=obs.chunk, t=obs.t, out_of_order=True))
        self.complete_stage(j, obs)
        k = self.current
        self.stage_interventions = 0
        self.begin_stage(k, obs, why=f"item {j + 1} done first")
        self.entry[k] = self.back_to = self.trace_index()

    def complete_stage(self, k, obs: Obs):
        s = self.stages[k]
        if not s.done_times or (s.undone_times and s.undone_times[-1]["t"] > s.done_times[-1]["t"]):
            s.done_times.append(dict(chunk=obs.chunk, t=obs.t))
            self.keyframes[f"s{k}_done"] = obs
        self.end_attempt(k, obs, "done")
        s.history = []

    def undo_stage(self, k, obs: Obs):
        """sub-goal k, done earlier, reads not done any more: it is current again"""
        self.stages[k].undone_times.append(dict(chunk=obs.chunk, t=obs.t))
        if self.current < len(self.stages) and self.current != k:
            self.end_attempt(self.current, obs, f"interrupted: sub-goal {k + 1} undone")
        self.current, self.stage_interventions = k, 0
        self.back_to = self.entry.get(k)               # loop.py: back_to = starts.get(s)

    def log_status(self, obs, status):
        self.status_log.append(dict(chunk=obs.chunk, t=obs.t, **to_json(status)))
        if status.done is not None:
            for k, d in enumerate(status.done[:len(self.stages)]):
                self.stages[k].last_status = dict(chunk=obs.chunk, t=obs.t, done=bool(d), source=status.source,
                                                  evidence=status.evidence if k == min(status.current, len(self.stages) - 1) else "")
        if status.note:
            self.add_note(obs, status.note, status.source)

    def add_note(self, obs, text, source, stage=None):
        self.notes.append(dict(chunk=obs.chunk, t=obs.t, stage=self.current if stage is None else stage,
                               source=source, text=str(text)[:300]))

    def add_dot(self, obs, k, progress, advancing, previous_ok=True, confidence=1.0):
        """a progress dot: the tracker's view of sub-goal k at this check"""
        self.dots.append(dict(chunk=obs.chunk, t=obs.t, stage=k, attempt=len(self.stages[k].attempts),
                              progress=progress, advancing=bool(advancing), previous_ok=bool(previous_ok),
                              confidence=float(confidence)))

    def attempt_dots(self, k=None):
        """the progress dots of sub-goal k's latest attempt (default: the current sub-goal)"""
        k = self.current if k is None else k
        n = len(self.stages[k].attempts) if k < len(self.stages) else 0
        return [d for d in self.dots if d["stage"] == k and d["attempt"] == n]

    def is_done(self, k):
        """sub-goal k recorded done (and not undone since)"""
        st = self.stages[k]
        return bool(st.done_times) and not (st.undone_times and st.undone_times[-1]["t"] > st.done_times[-1]["t"])

    def open_items(self, k=None):
        """the items judged now: sub-goal k (default: the current one) and, if it is any-order, the not-yet-done
        sub-goals of its any-order run after it"""
        k = self.current if k is None else k
        if k >= len(self.plan):
            return []
        out, j = [k], k + 1
        while self.plan[k].any_order and j < len(self.plan) and self.plan[j].any_order:
            if not self.is_done(j):
                out.append(j)
            j += 1
        return out

    def log_trigger(self, trig, obs):
        self.triggers.append(trig)
        self.keyframes[f"trigger_{obs.chunk}"] = obs

    def log_intervention(self, iv, obs, result=None):
        self.stages[iv.stage].interventions += 1
        self.interventions.append(dict(chunk=obs.chunk, t=obs.t, **to_json(iv), result=result or {}))

    def log_vlm(self, module, obs, res):
        self.vlm_calls.append(dict(module=module, chunk=obs.chunk, t=obs.t, **res.meta(),
                                   answer=res.data if res.data is not None else res.text[:500]))

    # ------------------------------------------------------------------ read by the modules
    def latest_start(self, k=None):
        k = self.current if k is None else k
        if k >= len(self.stages):
            return None
        a = self.stages[k].attempts
        return a[-1] if a else None

    def time_on_stage(self, obs):
        """GR00T's own time (env steps) since the current sub-goal's latest attempt began"""
        s = self.latest_start()
        return obs.t - s["t"] if s else obs.t

    def recent(self, n, stride=1):
        f = list(self.frames)
        return f[::-1][::stride][:n][::-1]

    def attempt_summary(self, k):
        """[dict(n, began_s, why, ended_s, how, seconds)] for sub-goal k, oldest first"""
        out = []
        for i, a in enumerate(self.stages[k].attempts):
            e = a["end"]
            out.append(dict(n=i + 1, began_s=round(a["t"] / CONTROL_HZ, 1), why=a["why"],
                            ended_s=None if e is None else round(e["t"] / CONTROL_HZ, 1),
                            how=None if e is None else e["how"],
                            seconds=None if e is None else round((e["t"] - a["t"]) / CONTROL_HZ, 1)))
        return out

    def gripper_events(self, since_trace_index=0):
        """[(trace index, "closed" | "opened")] from the robot trace's gripper commands since an entry"""
        if self.trace is None:
            return []
        p, out, prev = self.trace.path, [], None
        for i in range(max(1, since_trace_index), len(p)):
            g = p[i]["grip"]
            if g is None:
                continue
            g = g > 0.5
            if prev is not None and g != prev:
                out.append((i, "closed" if g else "opened"))
            prev = g
        return out

    def robot_state(self, obs, n_recent=4):
        """what the robot's own sensors say: gripper, finger gap, likely holding, motion recently and in this attempt"""
        gap = obs.finger_gap()
        closed = obs.gripper_cmd > 0.5
        out = dict(gripper="closed" if closed else "open", finger_gap_mm=None if gap is None else round(1000 * gap, 1))
        if gap is not None and closed:
            out["holding"] = "probably (the fingers stopped apart)" if gap >= EMPTY_GAP else "no (closed on nothing)"
        rec = self.recent(n_recent + 1)
        if len(rec) >= 2 and rec[0].hand() is not None:
            out["hand_moved_cm_recent"] = round(100 * float(np.linalg.norm(rec[-1].hand() - rec[0].hand())), 1)
            out["base_moved_cm_recent"] = round(100 * float(np.linalg.norm(rec[-1].base()[:2] - rec[0].base()[:2])), 1)
            out["recent_seconds"] = round((rec[-1].t - rec[0].t) / CONTROL_HZ, 1)
        a = self.latest_start()
        start = self.keyframes.get(f"s{self.current}_attempt")
        if a is not None and start is not None and start.hand() is not None:
            out["hand_from_attempt_start_cm"] = round(100 * float(np.linalg.norm(obs.hand() - start.hand())), 1)
            out["base_from_attempt_start_cm"] = round(100 * float(np.linalg.norm(obs.base()[:2] - start.base()[:2])), 1)
            ev = self.gripper_events(a["trace_index"] or 0) if a["trace_index"] is not None else []
            out["gripper_changes_this_attempt"] = [h for _, h in ev]
        return out

    def cost(self):
        return round(sum(c.get("cost_usd") or 0.0 for c in self.vlm_calls), 4)

    def summary(self):
        return dict(instruction=self.instruction, current=self.current, stages=[s.json() for s in self.stages],
                    expected=self.expected, notes=self.notes, triggers=[to_json(t) for t in self.triggers],
                    interventions=self.interventions, keyframes=sorted(self.keyframes), free_steps=self.free_steps,
                    vlm_calls=self.vlm_calls, vlm_cost_usd=self.cost(), status_checks=len(self.status_log),
                    plan=[dict(text=g.text, check=g.check, any_order=g.any_order) for g in self.plan],
                    dots=self.dots, moved_on=self.moved_on)
