"""Offline checks of the plain logic (no simulator, no GR00T server): python -I coordinator/tests/test_offline.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from coordinator.tests._util import obs  # noqa: E402

from coordinator.detection.timeout import StageTimeout
from coordinator.detection.undone import Undone
from coordinator.memory import Memory
from coordinator.motion import retrace
from coordinator.recovery.policies import Ladder, RetraceRetry
from coordinator.schema import StageStatus, SubGoal, Trigger, Intervention, to_json


def entry(q0, grip=0.0):
    return dict(q=np.full(7, q0), base=np.array([0.0, 0.0, 0.0]), rel=np.zeros(3), R=np.eye(3), grip=grip, mode=0.0)


def test_plan():
    # out 0.0 -> 0.3 rad, back to 0.0 (a loop), then a grasp at 0.2: the loop is cut, the grasp is kept
    path = [entry(0.0)] + [entry(0.02 * i) for i in range(1, 16)] + [entry(0.30 - 0.02 * i) for i in range(1, 16)]
    path += [entry(0.02 * i) for i in range(1, 11)] + [entry(0.20, grip=1.0) for _ in range(3)]
    seq = retrace.plan(path, 0)
    qs = [round(float(path[i]["q"][0]), 2) for i in seq]
    assert seq[0] == len(path) - 1 and seq[-1] == 0, seq
    assert max(qs) <= 0.2 + 1e-9, f"the loop out to 0.30 rad was not cut: {qs}"
    assert any(path[i]["grip"] == 0.0 for i in seq[1:]) and path[seq[0]]["grip"] == 1.0, "the gripper change is lost"
    print("retrace plan: ok", qs)


def mem(n=2):
    return Memory("task", [SubGoal(i, f"goal {i}") for i in range(n)], expected={0: 10, 1: 5})


def test_memory_and_timeout():
    m = mem()
    m.begin_stage(0, obs(0))
    to = StageTimeout("T", windows={("T", 0): 10, ("T", 1): 5})
    assert to.check(m, obs(9), None) == []
    t = to.check(m, obs(10), None)
    assert len(t) == 1 and t[0].kind == "timeout"
    assert to.check(m, obs(11), None) == [], "fired twice in one window"
    assert to.check(m, obs(20), None)[0].kind == "timeout", "a further window must fire again"
    m.end_attempt(0, obs(21), "timeout -> retrace")
    m.begin_stage(0, obs(21), why="retrace")
    assert m.time_on_stage(obs(22)) == 16
    assert to.check(m, obs(31), None)[0].kind == "timeout", "a new attempt must re-arm it"
    m.complete_stage(0, obs(33)); m.begin_stage(1, obs(33))
    assert m.current == 1 and m.stages[0].done_at["t"] == 33 * 16
    s = m.attempt_summary(0)
    assert [a["how"] for a in s] == ["timeout -> retrace", "done"] and s[1]["why"] == "retrace", s
    assert "s0_start" in m.keyframes and "s0_done" in m.keyframes and m.keyframes["s0_attempt"].chunk == 21
    print("memory and timeout: ok")


def test_undone():
    m = mem()
    m.begin_stage(0, obs(0)); m.complete_stage(0, obs(5)); m.begin_stage(1, obs(5))
    u = Undone()
    near = StageStatus(done=[False, False], current=0, source="t", hand_clear=False)
    clear = StageStatus(done=[False, False], current=0, source="t", hand_clear=True)
    unknown = StageStatus(done=[False, False], current=0, source="vlm", hand_clear=None)
    fine = StageStatus(done=[True, False], current=1, source="t", hand_clear=True)
    assert all(u.check(m, obs(6 + i), near) == [] for i in range(5)), "counted while the hand was near"
    assert u.check(m, obs(11), None) == [], "no status: nothing to count"
    assert u.check(m, obs(12), clear) == [] and u.check(m, obs(13), unknown) == []
    assert u.check(m, obs(14), clear)[0].kind == "undone"
    u.check(m, obs(15), clear); u.check(m, obs(16), fine)
    assert u.check(m, obs(17), clear) == [], "a done check must reset the count"
    m.undo_stage(0, obs(18))
    assert m.current == 0 and m.stages[1].attempts[-1]["end"]["how"].startswith("interrupted")
    m.begin_stage(0, obs(19), why="retrace"); m.complete_stage(0, obs(25))
    assert len(m.stages[0].done_times) == 2, "a redone sub-goal records its second completion"
    print("undone: ok")


def test_recovery():
    m = mem(1)
    m.begin_stage(0, obs(0))
    tg = Trigger("timeout", 0, 1)
    assert RetraceRetry().decide(m, obs(1), tg).kind == "retrace"
    lad, kinds = Ladder(steps=("resample", "reinstruct"), per_step=2, budget=3), []
    for _ in range(4):
        iv = lad.decide(m, obs(1), tg)
        kinds.append(iv.kind)
        if iv.kind != "stop":
            m.log_intervention(iv, obs(1))
    assert kinds == ["resample", "resample", "reinstruct", "stop"], kinds
    assert to_json(iv)["kind"] == "stop" and isinstance(to_json(Intervention("retrace", 0)), dict)
    print("recovery policies: ok")


def test_robot_state():
    m = mem()
    m.begin_stage(0, obs(0, hand=(0.3, 0, 0.6)))
    for c in range(1, 6):
        m.observe(obs(c, hand=(0.3 + 0.05 * c, 0, 0.6), gap=0.03, grip=1.0))
    r = m.robot_state(obs(6, hand=(0.6, 0, 0.6), gap=0.03, grip=1.0))
    assert r["gripper"] == "closed" and r["holding"].startswith("probably") and r["hand_from_attempt_start_cm"] == 30.0, r
    assert m.robot_state(obs(7, gap=0.001, grip=1.0))["holding"].startswith("no")
    print("robot state: ok")


if __name__ == "__main__":
    test_plan(); test_memory_and_timeout(); test_undone(); test_recovery(); test_robot_state()
    print("all offline checks passed")
