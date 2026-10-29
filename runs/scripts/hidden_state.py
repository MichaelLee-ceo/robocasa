"""State that set_state() does not restore: the robot controllers' internal values and MuJoCo's warm start.

set_state() (oracle.py) restores the flattened sim state (time, qpos, qvel, act) and the fixtures, but not:
- the gripper command ramp: PandaGripper.format_action moves current_action by 0.2 per call toward the commanded side;
- the part controllers' goals and caches: OSC goal_pos / goal_ori (accumulated in "desired" mode, i.e. base mode),
  goal_update_mode, origin pose, joint goals, cached kinematics and the new_update flag;
- the robot's buffers of recent values, the env's step counter and clock, and mjData's qacc_warmstart / ctrl;
- the observables' sampling clocks: each forced observation (rollout() makes one per start) advances them by one
  physics substep, so without a reset rollout r sampled its later observations r substeps earlier than rollout 0
  (found 2026-10-29: base_position 1.5 mm x r off).
Without these, a rollout started by set_state() inherits whatever the previous rollout left behind (found 2026-10-27:
identical start state + identical actions diverged at the first step in every pair).

capture(kenv) / restore(kenv, h): exact copy, for states reached by stepping (e.g. a search node, or the scene right
after load_scene, which is the main run's exact hidden state at t = 0).
settle(kenv, prev_state): a canonical hidden state for a state restored from a saved trajectory, where the true one is
unknown: every part controller's goal reset to the current pose, the gripper ramp inferred from the fingers
(gripper_ramp), warm start zeroed. Call restore(kenv, capture(kenv)) of that before each rollout from the same state,
so all of them start identically."""
import copy
import numpy as np

_PLAIN = (np.ndarray, np.generic, int, float, bool, str, type(None))
OPEN_OLD = abs(0.020833 * 2) * 1.9 - 0.004  # 75.2 mm: above it a gripper that touches no object is taken as opening
HELD_MARGIN = 0.0015  # a held object counts as gripped unless the gap is within 1.5 mm of fully open
FINGER_TRACK = 2.0  # free fingers cover about half the distance to the servo target per step: target ~ gap + 2 x (gap change)


def _attrs(obj):
    return {k: copy.deepcopy(v) for k, v in vars(obj).items() if isinstance(v, _PLAIN)}


def _set_attrs(obj, d):
    for k, v in d.items():
        setattr(obj, k, copy.deepcopy(v))


def capture(kenv):
    d = kenv.sim.data
    h = dict(warmstart=np.array(d.qacc_warmstart), ctrl=np.array(d.ctrl), qfrc_applied=np.array(d.qfrc_applied),
             xfrc_applied=np.array(d.xfrc_applied), timestep=kenv.timestep, cur_time=kenv.cur_time, robots=[],
             obs_clock={n: (o._time_since_last_sample, o._sampled, o._current_delay) for n, o in kenv._observables.items()})
    for r in kenv.robots:
        cc = r.composite_controller
        h["robots"].append(dict(
            composite=_attrs(cc),
            parts={n: _attrs(c) for n, c in cc.part_controllers.items()},
            grippers={n: _attrs(g) for n, g in r.gripper.items()},
            buffers={k: copy.deepcopy(v) for k, v in vars(r).items() if k.startswith("recent_")}))
    return h


def restore(kenv, h):
    d = kenv.sim.data
    d.qacc_warmstart[:] = h["warmstart"]; d.ctrl[:] = h["ctrl"]
    d.qfrc_applied[:] = h["qfrc_applied"]; d.xfrc_applied[:] = h["xfrc_applied"]
    kenv.timestep, kenv.cur_time = h["timestep"], h["cur_time"]
    for n, (ts, sampled, delay) in h.get("obs_clock", {}).items():
        o = kenv._observables[n]
        o._time_since_last_sample, o._sampled, o._current_delay = ts, sampled, delay
    for r, hr in zip(kenv.robots, h["robots"]):
        cc = r.composite_controller
        _set_attrs(cc, hr["composite"])
        for n, c in cc.part_controllers.items():
            _set_attrs(c, hr["parts"][n])
        for n, g in r.gripper.items():
            _set_attrs(g, hr["grippers"][n])
        for k, v in hr["buffers"].items():
            setattr(r, k, copy.deepcopy(v))


def finger_contacts(kenv, r, arm):
    """(free-jointed bodies touched by finger 1, by finger 2, fixture contacts pushing the fingers open, fixture contacts
    pushing them closed, any fixture contact). Robot bodies are ignored."""
    m, d = kenv.sim.model, kenv.sim.data
    g = r.gripper[arm]
    j1, j2 = (m.joint_name2id(j) for j in g.joints)
    b1, b2 = m.jnt_bodyid[j1], m.jnt_bodyid[j2]
    close = {b1: -np.array(d.xaxis[j1]), b2: np.array(d.xaxis[j2])}  # joint 1 opens with +q, joint 2 with -q

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
            if (m.body_id2name(m.body_rootid[ob]) or "").startswith(("robot", "gripper", "mobilebase", "mount")):
                continue
            root = m.body_rootid[ob]
            if m.body_jntadr[root] >= 0 and m.jnt_type[m.body_jntadr[root]] == 0:  # free joint: a movable object
                free[f].add(root)
            else:
                fixture = True
                if float(np.dot(fdir, close[f])) > 0:
                    n_close += 1
                else:
                    n_open += 1
    return free[b1], free[b2], n_open, n_close, fixture


def gripper_ramp(kenv, r, arm, prev_state=None):
    """the gripper ramp c = current_action[1] (+1 commanded closed, -1 open; 0.2 per step) that best explains the fingers.
    prev_state: the flattened sim state one step earlier on the saved trajectory.
      held (both fingers on the same object)  -> +1 unless the gap is within 1.5 mm of fully open
      an object and a fixture, not held       -> +1 iff gap < 75.2 mm and the gap did not widen by > 0.5 mm
      fixture only, fingers still (< 0.5 mm)  -> the contact side at a 5-75.2 mm gap (pushed open = something between
                                                 the fingers -> +1; squeezed shut, e.g. against a door -> -1), else as above
      otherwise (free fingers, or one object
      touched but not held, or moving)        -> the servo target: c = 1 - (gap + 2 x gap change) / 40 mm, on the 0.2 grid
    Checked 2026-10-29 against the true ramp at 77.5k chunk boundaries of saved rollouts (runs/20261029_review/restore):
    mean error 0.09 ramp steps and 0.5 % wrong side at saturated boundaries, vs 0.62 and 4.5 % for the +-1 rule it
    replaces. Mid-ramp states (grasp and release starts) are where +-1 failed most: 4.6 -> 0.8 steps."""
    m, d = kenv.sim.model, kenv.sim.data
    idx = r._ref_gripper_joint_pos_indexes[arm]
    q = d.qpos[idx]
    gap = abs(q[0] - q[1])
    dgap = 0.0
    if prev_state is not None:
        qp = prev_state[1 + np.asarray(idx)]
        dgap = gap - abs(qp[0] - qp[1])
    lo, hi = m.jnt_range[[m.joint_name2id(j) for j in r.gripper[arm].joints]].T
    open_gap = max(abs(hi[0] - lo[1]), abs(lo[0] - hi[1]))
    f1, f2, n_open, n_close, fixture = finger_contacts(kenv, r, arm)
    old = 1.0 if gap < OPEN_OLD and not dgap > 0.0005 else -1.0
    if f1 & f2:
        return 1.0 if gap < open_gap - HELD_MARGIN else -1.0
    if fixture and (f1 | f2):
        return old
    if fixture and abs(dgap) <= 0.0005:
        if 0.005 <= gap < OPEN_OLD and n_open != n_close:
            return 1.0 if n_open > n_close else -1.0
        return old
    return float(np.clip(np.round((1 - (gap + FINGER_TRACK * dgap) / 0.04) / 0.2) * 0.2, -1, 1))


def settle(kenv, prev_state=None):
    """canonical hidden state at the current sim state (see module docstring). prev_state: the flattened sim state one
    step earlier on the saved trajectory, used for the gripper's motion. Not for t = 0 of a saved episode: there the
    state right after load_scene is exact (the ramp is 0, which no rule infers); capture it instead."""
    kenv.sim.data.qacc_warmstart[:] = 0
    for r in kenv.robots:
        cc = r.composite_controller
        cc.update_state()
        for c in cc.part_controllers.values():
            c.new_update = True
            c.update(force=True)
            # The mobile base controllers turn every base command from the current base frame into the frame of the
            # base pose at the start of the episode (init_pos / init_ori), and reset_goal sets that pose to the current
            # one. The env holds the episode's start pose after load_scene; keep it. Overwriting it turned every base
            # command by the base's yaw since the start (2026-10-28: SteamInMicrowave s15 t512, turned about 180
            # degrees, every rollout drove away from the microwave).
            init = (c.init_pos, c.init_ori) if getattr(c, "init_ori", None) is not None else None
            c.reset_goal()
            if init is not None:
                c.init_pos, c.init_ori = init
            if getattr(c, "input_ref_frame", None) == "base" and hasattr(c, "world_to_origin_frame"):
                # OSC.reset_goal sets goal_pos / goal_ori in world coordinates, but with input_ref_frame "base" the goal
                # lives in the base (origin) frame. In arm mode ("achieved") the next set_goal recomputes it from the
                # current pose, so this goes unnoticed; in base mode ("desired") the next goal is the old goal + delta,
                # and the arm lunged towards a point offset by the base's world pose (2026-10-28: SteamInMicrowave
                # s13 t640, the bowl thrown to the floor in every rollout that started in base mode).
                c.goal_pos = c.world_to_origin_frame(np.array(c.ref_pos))
                c.goal_ori = c.goal_origin_to_eef_pose()[:3, :3]
        for arm, g in r.gripper.items():
            if not hasattr(g, "current_action"):
                continue
            # A +-1 ramp (rows up to settled-v4) was wrong at grasp and release starts, at t = 0 (true 0), at the slow end
            # of an opening (77-78.5 mm read as closing: fingers snapped shut by up to 44 mm), and for open fingers
            # squeezed against a fixture (the microwave door): runs/20261029_review/restore/report.md.
            c_ = gripper_ramp(kenv, r, arm, prev_state)
            g.current_action = np.array([-1.0, 1.0]) * c_
