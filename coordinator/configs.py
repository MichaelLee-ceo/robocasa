"""Named configurations: which implementation fills each slot. A run is (config, start, policy seed[, VLM spec]).

| config            | tracker | detectors                      | confirmer | recovery | runs now?                          |
|-------------------|---------|--------------------------------|-----------|----------|------------------------------------|
| plain             |         |                                |           |          | yes: GR00T alone                   |
| oracle_retrace    | oracle  | undone, timeout                | accept    | retrace  | yes: the current method            |
| oracle_retrace_ev | oracle  | + oracle events                | accept    | retrace  | yes: an earlier-detection bound    |
| oracle_vlm_recov  | oracle  | timeout, undone                | accept    | VLM      | yes: the recovery decision alone   |
| vlm_track         | VLM     | timeout, undone                | accept    | retrace  | yes: the tracker alone             |
| vlm_confirm       | VLM     | timeout, undone                | VLM       | retrace  | yes                                |
| vlm_full          | VLM     | timeout, undone                | VLM       | VLM      | yes: the proposal's coordinator    |
| proprio_triggers  | oracle  | timeout, undone, empty grasp,  | accept    | retrace  | after detection/proprio.py         |
|                   |         |   stationary                   |           |          |                                    |
| oracle_ladder     | oracle  | timeout, undone                | accept    | ladder   | after motion/backoff.py            |
| oracle_retrace_vlmdet | oracle | undone, VLM event, timeout   | accept    | retrace  | yes: VLM failure reports as early  |
|                   |         |                                |           |          |   triggers (detection/vlm_event.py)|
Detector order matters: the first accepted trigger is acted on, and loop.py handles an undone stage before a stuck
one, so "undone" comes first. VLM configs need `vlm` (a spec such as "claude:claude-opus-5-5", see vlm/__init__.py) and use the task's stage
sentences as the checklist (tracking.oracle.task_plan; plan="vlm" lets the VLM write it). The VLM tracker runs every
VLM_TRACK_EVERY chunks (cost: one call per check).
Every retrace config uses the final retrace (motion/retrace.py); a blocked retrace gets the unstick fallback
(recovery/unstick.py): the VLM picks GR00T's sentence when a `vlm` is given (any config), else the task sentence.
build(name, ep, port, tag, vlm=None, ...) -> loop.Coordinator (None for plain)."""
from . import context as C
from .detection.confirm import AcceptAll, VLMConfirmer
from .detection.oracle_events import OracleEvents
from .detection.proprio import EmptyGrasp, Stationary
from .detection.timeout import StageTimeout
from .detection.undone import Undone
from .detection.vlm_event import VLMEvent
from .executor import Executor
from .loop import Coordinator
from .paths import use_envkit
from .recovery.policies import Ladder, RetraceRetry
from .recovery.vlm import VLMRecovery
from .recovery.unstick import FixedUnstick, VLMUnstick
from .tracking.oracle import OracleTracker, task_plan
from .tracking.vlm import VLMTracker
from .vlm import make_vlm

use_envkit()
import envkit  # noqa: E402

VLM_TRACK_EVERY = 4
CONFIGS = {
    "plain": None,
    "oracle_retrace": dict(tracker="oracle", detectors=["undone", "timeout"], confirmer="accept", recovery="retrace"),
    "oracle_retrace_ev": dict(tracker="oracle", detectors=["undone", "timeout", "oracle_events"], confirmer="accept",
                              recovery="retrace"),
    "oracle_vlm_recov": dict(tracker="oracle", detectors=["undone", "timeout"], confirmer="accept", recovery="vlm"),
    "vlm_track": dict(tracker="vlm", detectors=["undone", "timeout"], confirmer="accept", recovery="retrace"),
    "vlm_confirm": dict(tracker="vlm", detectors=["undone", "timeout"], confirmer="vlm", recovery="retrace"),
    "vlm_full": dict(tracker="vlm", detectors=["undone", "timeout"], confirmer="vlm", recovery="vlm"),
    "proprio_triggers": dict(tracker="oracle", detectors=["undone", "timeout", "empty_grasp", "stationary"],
                             confirmer="accept", recovery="retrace"),
    "oracle_ladder": dict(tracker="oracle", detectors=["undone", "timeout"], confirmer="accept", recovery="ladder"),
    "oracle_retrace_vlmdet": dict(tracker="oracle", detectors=["undone", "vlm_event", "timeout"], confirmer="accept",
                                  recovery="retrace"),
}


def needs_vlm(name):
    c = CONFIGS[name]
    return c is not None and ("vlm" in (c["tracker"], c["confirmer"], c["recovery"]) or "vlm_event" in c["detectors"])


def build(name, ep, port, tag, vlm=None, context=None, budget=None, track_every=None, plan="given", free_retrace=True,
          event_n=2, event_every=4):
    cfg = CONFIGS[name]
    if cfg is None:
        return None
    if needs_vlm(name) and vlm is None:
        raise ValueError(f"config {name} needs a VLM (e.g. vlm='claude:claude-opus-5-5')")
    ctx = context or C.ContextConfig()
    if needs_vlm(name):
        vlm = make_vlm(vlm)   # one backend shared by the modules
    timeout = StageTimeout(ep.task)
    if cfg["tracker"] == "oracle":
        tracker = OracleTracker(ep)
    else:
        tracker = VLMTracker(vlm, ctx, plan=task_plan(ep) if plan == "given" else None)
    make = {"timeout": lambda: timeout, "undone": Undone, "empty_grasp": EmptyGrasp, "stationary": Stationary,
            "oracle_events": lambda: OracleEvents(ep), "vlm_event": lambda: VLMEvent(vlm, ctx, every=event_every, n=event_n)}
    dets = [make[d]() for d in cfg["detectors"]]
    confirmer = AcceptAll() if cfg["confirmer"] == "accept" else VLMConfirmer(vlm, ctx)
    recovery = {"retrace": lambda: RetraceRetry(budget), "ladder": lambda: Ladder(),
                "vlm": lambda: VLMRecovery(vlm, ctx, budget=budget)}[cfg["recovery"]]()
    every = track_every or (VLM_TRACK_EVERY if cfg["tracker"] == "vlm" else 1)
    n_stages = len(task_plan(ep))
    expected = {k: timeout.window(k) for k in range(n_stages)}
    return Coordinator(ep, Executor(port, tag), tracker, dets, confirmer, recovery, envkit.get_task_horizon(ep.task),
                       free_retrace=free_retrace, track_every=every, expected=expected,
                       unstick=VLMUnstick(vlm) if vlm is not None else FixedUnstick(None))
