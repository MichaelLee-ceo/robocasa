"""State that set_state() does not restore: the robot controllers' internal values and MuJoCo's warm start.

set_state() (oracle.py) restores the flattened sim state (time, qpos, qvel, act) and the fixtures, but not:
- the gripper command ramp: PandaGripper.format_action moves current_action by 0.2 per call toward the commanded side;
- the part controllers' goals and caches: OSC goal_pos / goal_ori (accumulated in "desired" mode, i.e. base mode),
  goal_update_mode, origin pose, joint goals, cached kinematics and the new_update flag;
- the robot's buffers of recent values, the env's step counter and clock, and mjData's qacc_warmstart / ctrl.
Without these, a rollout started by set_state() inherits whatever the previous rollout left behind (found 2026-10-27:
identical start state + identical actions diverged at the first step in every pair).

capture(kenv) / restore(kenv, h): exact copy, for states reached by stepping (e.g. a search node).
settle(kenv, prev_state): a canonical hidden state for a state restored from a saved trajectory, where the true one is
unknown: every part controller's goal reset to the current pose, the gripper ramp set to the commanded side inferred from
the fingers (closed or closing -> close), warm start zeroed. Call restore(kenv, capture(kenv)) of that before each
rollout from the same state, so all of them start identically."""
import copy
import numpy as np

_PLAIN = (np.ndarray, np.generic, int, float, bool, str, type(None))


def _attrs(obj):
    return {k: copy.deepcopy(v) for k, v in vars(obj).items() if isinstance(v, _PLAIN)}


def _set_attrs(obj, d):
    for k, v in d.items():
        setattr(obj, k, copy.deepcopy(v))


def capture(kenv):
    d = kenv.sim.data
    h = dict(warmstart=np.array(d.qacc_warmstart), ctrl=np.array(d.ctrl), qfrc_applied=np.array(d.qfrc_applied),
             xfrc_applied=np.array(d.xfrc_applied), timestep=kenv.timestep, cur_time=kenv.cur_time, robots=[])
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
    for r, hr in zip(kenv.robots, h["robots"]):
        cc = r.composite_controller
        _set_attrs(cc, hr["composite"])
        for n, c in cc.part_controllers.items():
            _set_attrs(c, hr["parts"][n])
        for n, g in r.gripper.items():
            _set_attrs(g, hr["grippers"][n])
        for k, v in hr["buffers"].items():
            setattr(r, k, copy.deepcopy(v))


def settle(kenv, prev_state=None):
    """canonical hidden state at the current sim state (see module docstring). prev_state: the flattened sim state one
    step earlier on the saved trajectory, used to tell an opening gripper from a closing one."""
    kenv.sim.data.qacc_warmstart[:] = 0
    for r in kenv.robots:
        cc = r.composite_controller
        cc.update_state()
        for c in cc.part_controllers.values():
            c.new_update = True
            c.update(force=True)
            c.reset_goal()
        for arm, g in r.gripper.items():
            if not hasattr(g, "current_action"):
                continue
            q = kenv.sim.data.qpos[r._ref_gripper_joint_pos_indexes[arm]]
            gap = abs(q[0] - q[1])
            open_gap = abs(g.init_qpos[0] - g.init_qpos[1]) * 1.9  # fully open is about twice the init gap
            closing = gap < open_gap - 0.004
            if prev_state is not None and closing:
                kenv_q_prev = prev_state[1 + np.asarray(r._ref_gripper_joint_pos_indexes[arm])]
                closing = not (gap - abs(kenv_q_prev[0] - kenv_q_prev[1]) > 0.0005)  # widening -> it was told to open
            g.current_action = np.array([-1.0, 1.0]) * (1.0 if closing else -1.0)
