"""The coordinator loop (proposal Figure 1): GR00T performs every action; the coordinator keeps the memory and decides
when to intervene.

At every chunk end:
  1. observe   memory.observe(Obs): the three views, the proprio, GR00T's own time, the trace entry
  2. track     every `track_every` chunks: tracker.update -> StageStatus; sub-goals that became done are closed in the
               memory and the next one begins (its attempt records the trace entry a retrace goes back to)
  3. detect    every detector's triggers (status = this chunk's StageStatus, or None)
  4. confirm   the confirmer accepts or rejects each trigger (all logged); an accepted "undone" makes that sub-goal
               current again
  5. recover   for the first accepted trigger: recovery.decide -> Intervention -> execute; the attempt it ends is
               closed in the memory with how it ended ("timeout -> retrace")
  6. act       GR00T's next chunk (memory.gr00t_instruction, memory.stream), unless the run is over
The run ends at the official success (at a chunk end, within the official horizon on GR00T's own time), when GR00T's
time reaches the horizon, or when the env ends. With free_retrace the env runs with a free horizon and retrace steps are
left out of GR00T's time (a retrace stands for a restart, the user's choice, 2026-11-06); a success reached while
retracing counts and is flagged.

Starting points: run() from the episode start, or replay(recorded chunks) first and pass a start trigger (the phase 0
failure points: detection assumed solved there)."""
from .memory import Memory
from .motion import retrace as rtrace
from .schema import Obs


class Coordinator:
    def __init__(self, ep, executor, tracker, detectors, confirmer, recovery, horizon, free_retrace=True,
                 track_every=1, retrace_max=400, expected=None):
        self.ep, self.executor, self.tracker = ep, executor, tracker
        self.detectors, self.confirmer, self.recovery = list(detectors), confirmer, recovery
        self.H, self.free, self.track_every, self.retrace_max = horizon, free_retrace, track_every, retrace_max
        self.expected = expected or {}
        self.trace = rtrace.RobotTrace(ep)  # attached before any step: the whole path is recorded
        self.memory = None
        self.success_t, self.success_in_retrace, self.intervening, self.n_track = None, False, True, 0
        self.uses_sim = sorted({type(x).__name__ for x in [tracker, *self.detectors, confirmer, recovery]
                                if getattr(x, "uses_sim", False)})

    # ------------------------------------------------------------------ clock and observation
    def ptime(self):
        return self.ep.t - (self.memory.free_steps if self.free and self.memory is not None else 0)

    def obs(self):
        g = float(self.ep.chunks[-1]["actions"]["action.gripper_close"][-1, 0]) if self.ep.chunks else 0.0
        return Obs.from_env(self.ep.obs, len(self.ep.chunks), self.ptime(), g, trace_index=len(self.trace.path) - 1)

    def done(self):
        if self.free:
            return self.success_t is not None or self.ptime() >= self.H or self.ep.done
        return self.ep.success or self.ep.done

    def _chunk_success(self, c):
        """official success at this GR00T chunk's end; a chunk past H ends at H, as the official eval cuts it"""
        e = min(c["n"], self.H - (c["t"] - (self.memory.free_steps if self.free else 0))) - 1
        return e >= 0 and bool(c["step_success"][e])

    def _observe(self, o):
        if not self.memory.frames or self.memory.frames[-1].chunk != o.chunk:
            self.memory.observe(o)

    # ------------------------------------------------------------------ setup
    def start(self):
        o = self.obs()
        plan = self.tracker.make_plan(self.ep.instruction, o)
        self.memory = Memory(self.ep.instruction, plan, self.trace, self.expected)
        if getattr(self.tracker, "plan_call", None) is not None:
            self.memory.log_vlm("plan", o, self.tracker.plan_call)
        self.memory.begin_stage(0, o)
        self._observe(o)

    def replay(self, chunks, stage_after=None, cap=None):
        """execute recorded chunks exactly (no policy), watching as in a live run. stage_after[i] = the sub-goal after
        i chunks (the recorded tracker, before chunk i runs), used to record where each sub-goal began (for a later
        retrace), up to sub-goal `cap`"""
        m = self.memory
        top = len(m.plan) - 1 if cap is None else min(cap, len(m.plan) - 1)
        for i, c in enumerate(chunks):
            if stage_after is not None:
                o = self.obs()
                while m.current < min(stage_after[i], top):
                    m.complete_stage(m.current, o)
                    m.begin_stage(m.current + 1, o, why="replay")
            self.ep.step(c["actions"])
            self._observe(self.obs())

    # ------------------------------------------------------------------ one intervention
    def execute(self, iv, obs):
        m, out = self.memory, {}
        if iv.kind == "stop":
            self.intervening = False
        elif iv.kind == "continue":
            pass
        elif iv.kind == "retrace":
            target = m.back_to if m.back_to is not None else m.latest_start(iv.stage)["trace_index"]
            n0 = len(self.ep.chunks)
            out = rtrace.retrace(self.ep, self.trace, target, self.retrace_max)
            m.free_steps += out["steps"]
            if any(c["step_success"][-1] for c in self.ep.chunks[n0:]):
                self.success_t, self.success_in_retrace = self.ptime(), True
            if out["arrived"]:                               # loop.py: back_to = len(rec_path.path) - 1
                m.back_to = m.trace_index()
            o = self.obs()
            m.begin_stage(iv.stage, o, why="retrace", trace_index=m.back_to)
            self._observe(o)
        elif iv.kind in ("resample", "reinstruct"):
            m.begin_stage(iv.stage, obs, why=iv.kind, trace_index=m.latest_start(iv.stage)["trace_index"])
        elif iv.kind == "backoff":
            raise NotImplementedError("backoff: see motion/backoff.py")
        else:
            raise ValueError(iv.kind)
        if iv.kind in ("retrace", "resample", "reinstruct", "backoff"):
            m.stage_interventions += 1                       # loop.py: attempts += 1; stream f"{tag}/a{cur}.{attempts}"
            m.stream = f"a{iv.stage}.{m.stage_interventions}"
            m.gr00t_instruction = iv.instruction
        m.log_intervention(iv, obs, out)

    # ------------------------------------------------------------------ the loop
    def run(self, start_trigger=None):
        m = self.memory
        pending = [start_trigger] if start_trigger else []
        while not self.done():
            o = self.obs()
            self._observe(o)
            status = None
            if self.n_track % self.track_every == 0:
                status = self.tracker.update(m, o)
                m.log_status(o, status)
                if status.done is not None and status.current > m.current and not pending:   # moved on
                    m.move_on(status.current, o)
            self.n_track += 1
            trigs, pending = list(pending), []
            for d in self.detectors:
                trigs += d.check(m, o, status)
            for tg in trigs:
                ok, why = self.confirmer.confirm(m, o, tg)
                tg.confirmed, tg.reason = bool(ok), why
                m.log_trigger(tg, o)
                if not ok:
                    continue
                if tg.kind == "undone":
                    m.undo_stage(tg.stage, o)
                if self.intervening:
                    iv = self.recovery.decide(m, o, tg)
                    if iv.kind not in ("continue", "stop"):
                        m.end_attempt(iv.stage, o, f"{tg.kind} -> {iv.kind}")
                    self.execute(iv, o)
                    break
            if self.done():
                break
            self.ep.step(self.executor.act(self.ep.obs, m, len(self.ep.chunks)))
            if self.free and self.success_t is None and self._chunk_success(self.ep.chunks[-1]):
                self.success_t = min(self.ptime(), self.H)
        return self.result()

    def result(self):
        m = self.memory
        success = (self.success_t is not None) if self.free else bool(self.ep.success)
        o = self.obs()
        if success:  # the last sub-goal's check is the official success: every sub-goal is done now
            for k in range(len(m.plan)):
                if m.stages[k].open_attempt() is not None or not m.stages[k].done_times:
                    m.complete_stage(k, o)
        elif m.current < len(m.plan):
            m.end_attempt(m.current, o, "episode end")
        return dict(success=success, success_t=self.success_t if self.free else self.ep.success_step,
                    success_in_retrace=self.success_in_retrace, gr00t_time=self.ptime(), horizon=self.H,
                    chunks=len(self.ep.chunks), free_steps=m.free_steps, uses_sim=self.uses_sim,
                    vlm_calls=len(m.vlm_calls), vlm_cost_usd=m.cost(), memory=m.summary())
