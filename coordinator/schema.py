"""The records passed between the coordinator's modules. Plain data, so every record can be logged as JSON (images
excepted) and a module can be swapped without touching the others.

Rule: everything here is what a real robot could have. Simulator ground truth never goes into these records; modules
that read the simulator (the oracle ones) get the episode at construction and say so with `uses_sim = True`."""
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

CONTROL_HZ = 20
STEPS_PER_CHUNK = 16
CAMERAS = {"video.robot0_agentview_left": "left shoulder view", "video.robot0_agentview_right": "right shoulder view",
           "video.robot0_eye_in_hand": "wrist camera"}
PROPRIO = ("state.base_position", "state.base_rotation", "state.end_effector_position_relative",
           "state.end_effector_rotation_relative", "state.gripper_qpos")


@dataclass
class Obs:
    """What the coordinator sees at a chunk end: the policy's own observation, plus bookkeeping."""
    chunk: int                      # chunks executed in the env so far (retrace chunks included)
    t: int                          # GR00T's own time in env steps (retrace steps not counted)
    images: dict                    # camera key -> (256, 256, 3) uint8: the three views GR00T sees
    proprio: dict                   # state.* key -> 1-D float array (base pose, hand pose relative to the base, fingers)
    gripper_cmd: float              # the last gripper command sent (1 = close, 0 = open)
    trace_index: Optional[int] = None   # the robot trace's entry at this moment (memory.trace)

    @classmethod
    def from_env(cls, ep_obs, chunk, t, gripper_cmd, trace_index=None):
        return cls(chunk=chunk, t=t, images={c: np.asarray(ep_obs[c])[0] for c in CAMERAS},
                   proprio={k: np.asarray(ep_obs[k], float).reshape(-1) for k in PROPRIO},
                   gripper_cmd=float(gripper_cmd), trace_index=trace_index)

    @property
    def seconds(self):
        return self.t / CONTROL_HZ

    def finger_gap(self):
        """distance between the fingers (m): about 0.04 open, under 0.004 closed on nothing"""
        q = self.proprio.get("state.gripper_qpos")
        return float(abs(q[0]) + abs(q[1])) if q is not None and len(q) >= 2 else None

    def hand(self):
        """hand position relative to the base (m)"""
        return self.proprio.get("state.end_effector_position_relative")

    def base(self):
        """base position (m, world)"""
        return self.proprio.get("state.base_position")


@dataclass
class SubGoal:
    """One entry of the task checklist."""
    index: int
    text: str                       # instruction-like sentence (the atomic task it corresponds to)
    check: str = ""                 # checkable completion condition in words, e.g. "the kettle is on a burner"
    any_order: bool = False         # with the neighbouring any_order sub-goals, may be done in any order


@dataclass
class StageStatus:
    """The tracker's answer at one moment. done = None: no answer this time (e.g. the VLM call failed)."""
    done: Optional[list]            # bool per sub-goal, judged now
    current: int                    # the first sub-goal not done (len(done) when all are done)
    source: str                     # "oracle" | "vlm" | ...
    confidence: float = 1.0
    evidence: str = ""              # what supports the answer (VLM: what it saw)
    hand_clear: Optional[bool] = None  # the hand touches nothing and is away from all objects (None: unknown)
    note: str = ""                  # something the tracker wants remembered (goes to memory.notes)
    error: str = ""
    extra_done: list = field(default_factory=list)  # sub-goals after the current one ticked now, out of order
                                                    # (an any-order run: memory.tick_out_of_order)


@dataclass
class Trigger:
    """A detector's claim that something may be wrong."""
    kind: str                       # "failure_point" | "timeout" | "undone" | "empty_grasp" | "stationary" | event name
    stage: int                      # the sub-goal it concerns
    chunk: int
    detail: dict = field(default_factory=dict)
    confirmed: Optional[bool] = None
    reason: str = ""                # the confirmer's reason


@dataclass
class Intervention:
    """What the recovery policy decided."""
    kind: str                       # "continue" | "resample" | "retrace" | "reinstruct" | "backoff" | "stop"
    stage: int
    instruction: Optional[str] = None   # the sentence GR00T gets from now on (None: the task sentence)
    reason: str = ""
    detail: dict = field(default_factory=dict)


def to_json(x):
    """a record as JSON-able data (images dropped)"""
    d = asdict(x) if hasattr(x, "__dataclass_fields__") else dict(x)
    d.pop("images", None)
    d.pop("proprio", None)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
