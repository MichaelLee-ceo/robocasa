"""Candidate replacement for the gripper part of hidden_state.settle (NOT applied; review 2026-10-29).

Rule "g" (out/grip_eval_geo.txt + the held variant in the report): wrong side at 0.63 % of 74k saturated chunk
boundaries (current v4 rule 4.5 %, old v3 rule 1.6 %, trend rule v5 4.7 %, value rule 6.1 %), mean ramp error 0.24 policy
steps (v4: 0.62). Per task: KettleBoiling 0.64 % (v4 3.9 %), SteamInMicrowave 1.7 % (8.6 %), PackIdenticalLunches 0.08 % (2.6 %).
  held (both fingers touch the same free-jointed body)  -> closing unless the gap is within 1.5 mm of fully open (the v4
                                                         threshold; no widening test: a held object often pushes the
                                                         fingers apart by > 0.5 mm in one step, e.g. SteamInMicrowave
                                                         s1 t544 c6 b96, where reading it as opening dropped the bowl)
  touching a fixture only (no free body), gap 5-75.2 mm  -> contact side: the contacts push the fingers open (something
                                                         between them: knob, handle) -> closing; they push the fingers
                                                         closed (an open gripper squeezed against a door) -> opening
  otherwise                                              -> the v3 rule: closing iff gap < 75.2 mm and not widened > 0.5 mm
And t = 0: do not settle at all; the hidden state right after load_scene + replay(t=0) is the main run's exactly
(robot.reset() leaves current_action = 0; settle set it to +1 and every t = 0 rollout closed the gripper 41 -> 24 mm).

Usage in hidden_state.settle (sketch):
    for arm, g in r.gripper.items():
        closing = gripper_closing(kenv, r, arm, prev_state)
        g.current_action = np.array([-1.0, 1.0]) * (1.0 if closing else -1.0)
and in collect.decision_point:
    if t == 0: H0 = hidden_state.capture(kenv)          # exact
    else: hidden_state.settle(kenv, ep_states[t - 1]); H0 = hidden_state.capture(kenv)"""
import numpy as np

OPEN_V3 = abs(0.020833 - (-0.020833)) * 1.9 - 0.004   # 75.17 mm (old rule)
OPEN_MARGIN = 0.0015                                    # v4: within 1.5 mm of the joint-limit gap


def _free_root(m, body):
    root = m.body_rootid[body]
    j = m.body_jntadr[root]
    return root if (j >= 0 and m.jnt_type[j] == 0) else None


def finger_contacts(kenv, r, arm):
    """(free bodies touched by finger 1, free bodies touched by finger 2, fixture contacts pushing the fingers open,
    fixture contacts pushing them closed, any fixture contact)."""
    m, d = kenv.sim.model, kenv.sim.data
    g = r.gripper[arm]
    j1, j2 = (m.joint_name2id(j) for j in g.joints)
    b1, b2 = m.jnt_bodyid[j1], m.jnt_bodyid[j2]
    close = {b1: -np.array(d.xaxis[j1]), b2: np.array(d.xaxis[j2])}   # joint1 opens with +q, joint2 with -q

    def finger_of(body):  # the finger body or one of its children
        while body > 0:
            if body in close:
                return body
            body = m.body_parentid[body]
        return None
    robot_roots = {m.body_rootid[m.jnt_bodyid[j]] for j in r._ref_joint_indexes} if hasattr(r, "_ref_joint_indexes") else set()
    free = {b1: set(), b2: set()}
    n_open = n_close = 0
    fixture = False
    for i in range(d.ncon):
        c = d.contact[i]
        ga, gb = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        n = np.array(c.frame[:3])
        for fb, ob, fdir in ((ga, gb, -n), (gb, ga, n)):
            f = finger_of(fb)
            if f is None or finger_of(ob) is not None or m.body_rootid[ob] in robot_roots:
                continue
            name = m.body_id2name(m.body_rootid[ob]) or ""
            if name.startswith(("robot", "gripper", "mobilebase", "mount")):
                continue
            fr = _free_root(m, ob)
            if fr is not None:
                free[f].add(fr)
            else:
                fixture = True
                if float(np.dot(fdir, close[f])) > 0:
                    n_close += 1
                else:
                    n_open += 1
    return free[b1], free[b2], n_open, n_close, fixture


def gripper_closing(kenv, r, arm, prev_state=None):
    """True if the gripper should be restored as commanded closed (ramp +1), False for open (-1)."""
    m, d = kenv.sim.model, kenv.sim.data
    idx = r._ref_gripper_joint_pos_indexes[arm]
    q = d.qpos[idx]
    gap = abs(q[0] - q[1])
    widening = False
    if prev_state is not None:
        qp = prev_state[1 + np.asarray(idx)]
        widening = gap - abs(qp[0] - qp[1]) > 0.0005
    g = r.gripper[arm]
    lo, hi = m.jnt_range[[m.joint_name2id(j) for j in g.joints]].T
    open_gap = max(abs(hi[0] - lo[1]), abs(lo[0] - hi[1]))
    f1, f2, n_open, n_close, fixture = finger_contacts(kenv, r, arm)
    if f1 & f2:                                   # held between both fingers
        return gap < open_gap - OPEN_MARGIN
    if fixture and not (f1 | f2) and 0.005 <= gap < OPEN_V3 and n_open != n_close:
        return n_open > n_close                   # pushed open = something between the fingers
    return gap < OPEN_V3 and not widening
