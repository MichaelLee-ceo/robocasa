"""The robot's own trace, and the retrace: a rewind of it to where a sub-goal began.

The user's design: the stage is the unit; to retry it the robot goes back the way it came, taking what it holds with
it. Nothing in the world is set (no restore, no teleport): objects move only by contact, what fell stays.

RobotTrace(ep): wraps envkit's per-step log (changes nothing) and pushes, after every env step, the state of every part
GR00T moves: the arm's 7 joints and their speeds (its end-effector commands act through the OSC), the base (world x,
y, yaw), the torso height, the gripper command, and the OSC goal relative to the hand (control_mode decides how that
goal is updated); and the hand's world position. Entry 0 is the state when it was attached. While `override` is set,
it is called before each env step with that step's action (a view into the chunk, so changes go into the record).

plan(path, start): the entries from the last one back to `start`: the trace is a stack, popped one entry per step,
so the robot goes back exactly the way it came.

retrace(ep, trace, start): one popped entry per step (the forward speed), closed loop at every env step:
- arm: joint-space impedance toward the entry's joints and its recorded joint speeds reversed (torque = M (KP e +
  KD (qdot_ref - qdot)) + gravity and Coriolis, KP 150, critically damped; MuJoCo clamps it to the same actuator limits
  as the OSC), in place of the OSC;
- base: the command that takes it to the entry (the recorded step reversed, plus what it lags); torso: toward the
  entry's height;
- gripper: the command in effect at that entry (it opens where it had closed), holding still SETTLE steps before and
  after each change;
- the robot lags its entry by more than LAG_Q rad (arm) or LAG_B m / LAG_Y rad (base): the playback waits for it; a lag
  that does not shrink while waiting STUCK_STEPS steps means it cannot go back this way (e.g. the arm wedged against a
  fixture GR00T had pushed it into): the retrace stops `blocked` (the coordinator then tries unstick, recovery/
  unstick.py, and resumes with `end` = resume_entry(..));
- arrived (joints within 0.03 rad, base within 1.5 cm and 0.02 rad of the start entry): held there until the arm is at
  rest, and the OSC goal is set at the start entry's offset from the hand.
Its env steps are reported (`steps`) so that the caller can leave them out of GR00T's time.

Not reproducible by commands: the gripper's command ramp at the episode start (0 after reset; gripper_close is binary,
the ramp rests only at +-1). Checks and results: coordinator/tests/check_retrace_state.py (47 failure points: 45 back
at the stage start, hand within 0.5 mm median, 1.3 mm p90; duration 1.08 x the forward one), check_retrace_fallback.py;
runs/20261106_coordinator/README.md."""
import numpy as np

from ..schema import STEPS_PER_CHUNK

BASE_SPEED = np.array([0.030, 0.029, 0.066])   # base displacement per unit command per step: x, y (m), yaw (rad)
KP_J = 150.0
KD_J = 2 * np.sqrt(KP_J)
TORSO_K = 20.0                                  # torso command per m of height error (1.0 moves about 2 cm per step)
LAG_Q, LAG_B, LAG_Y = 0.25, 0.15, 0.3           # the robot lags its entry (arm rad, base m, base rad): the playback waits
STUCK_STEPS = 64                                # a lag that does not shrink by a fifth over this many steps: blocked
SETTLE = 8                                      # steps held still before and after a gripper change
BACK_TOL = (0.03, 0.015, 0.02)                  # arrived: joints rad, base m, base rad
REST, FINAL_MAX = 0.02, 4                       # at rest: arm joint speeds below REST rad/s (at most FINAL_MAX chunks)


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RobotTrace:
    def __init__(self, ep):
        e = self.e = ep.kenv
        m = e.sim.model
        self.ctrl = e.robots[0].part_controllers["right"]
        self.hand = e.robots[0].eef_site_id["right"]
        self.base = m.body_name2id("mobilebase0_base")
        self.arm = np.array([m.jnt_qposadr[m.joint_name2id(f"robot0_joint{i}")] for i in range(1, 8)])
        self.arm_dof = np.array([m.jnt_dofadr[m.joint_name2id(f"robot0_joint{i}")] for i in range(1, 8)])
        self.torso = m.jnt_qposadr[m.joint_name2id("mobilebase0_joint_torso_height")]
        self.override = None
        self.path = [self.pose(None)]
        orig = ep._log.step

        def step(action):
            if self.override is not None:
                self.override(action)
            out = orig(action)
            self.path.append(self.pose(action))
            return out

        ep._log.step = step

    def pose(self, action):
        d = self.e.sim.data
        b, Rb = d.xpos[self.base], d.xmat[self.base].reshape(3, 3)
        g = None if action is None else float(np.asarray(action["action.gripper_close"]).ravel()[0])
        return dict(q=d.qpos[self.arm].copy(), qd=d.qvel[self.arm_dof].copy(), base=np.array([b[0], b[1], _yaw(Rb)]),
                    torso=float(d.qpos[self.torso]), grip=g, goal=self.goal(),
                    hand_w=np.array(d.site_xpos[self.hand], dtype=float))

    def goal(self):
        """the OSC goal relative to the hand (offset in the OSC's frame, rotation hand -> goal); None before the first
        control update"""
        c = self.ctrl
        try:
            ach = c.world_to_origin_frame(np.array(c.ref_pos))
            R = c.goal_origin_to_eef_pose()[:3, :3]
            return np.asarray(c.goal_pos, dtype=float) - ach, R.T @ np.asarray(c.goal_ori, dtype=float)
        except Exception:
            return None


def grip_at(path, i):
    """the gripper command in effect at entry i (the one of the step that led to it; open before any step)"""
    while i > 0 and path[i]["grip"] is None:
        i -= 1
    return path[i]["grip"] if path[i]["grip"] is not None else 0.0


def _gap(a, b):
    """(max joint diff, base distance, base yaw diff) between two entries"""
    return (float(np.abs(a["q"] - b["q"]).max()), float(np.linalg.norm(a["base"][:2] - b["base"][:2])),
            abs(float(_wrap(a["base"][2] - b["base"][2]))))


def plan(path, start):
    """the entries to play back: every one, from the last to `start` (the trace is a stack, popped one per step)"""
    return list(range(len(path) - 1, start - 1, -1))


def resume_entry(path, start, blocked_entry, stuck_hand, clear=0.25):
    """after a detour off a blocked retrace: the last entry before the blocked one (towards `start`) at which the hand
    was `clear` m or more from where it got stuck (the retrace resumes there); `start` if there is none"""
    for e in plan(path, start):
        if e <= blocked_entry and np.linalg.norm(path[e]["hand_w"] - stuck_hand) >= clear:
            return e
    return start


def retrace(ep, rec, start, max_chunks=400, on_chunk=None, end=None):
    """rewind the path of RobotTrace `rec` to entry `start` (see the module doc); `end`: play back from this entry
    instead of the last one (resuming after a detour). Returns a summary dict."""
    path = list(rec.path) if end is None else list(rec.path)[:end + 1]
    seq = plan(path, start)
    ctrl, d = rec.ctrl, rec.e.sim.data
    st = dict(k=0, k_prev=None, settle=0, pending=None, grip=grip_at(path, len(path) - 1), qt=path[seq[0]]["q"].copy(),
              qd=np.zeros(7), grip_changes=0, final=False, wait=0, waited=0, size0=0.0, blocked=False)

    def run_controller():  # joint-space impedance toward st["qt"] at st["qd"], in place of the OSC
        ctrl.update()
        acc = KP_J * (st["qt"] - ctrl.joint_pos) + KD_J * (st["qd"] - ctrl.joint_vel)
        ctrl.torques = ctrl.mass_matrix @ acc + ctrl.torque_compensation
        ctrl.new_update = True
        return ctrl.torques

    def override(action):
        now = rec.pose(None)
        if st["settle"]:
            st["settle"] -= 1
            if st["settle"] == SETTLE and st["pending"] is not None:
                st["grip"], st["pending"] = st["pending"], None
        elif st["k"] + 1 < len(seq):           # pop the next entry, unless the robot lags far behind this one
            w = path[seq[st["k"]]]
            la, lb = float(np.abs(now["q"] - w["q"]).max()), float(np.linalg.norm(now["base"][:2] - w["base"][:2]))
            size = max(la / LAG_Q, lb / LAG_B, abs(float(_wrap(now["base"][2] - w["base"][2]))) / LAG_Y)
            if size > 1:                       # wait for it; blocked if the lag does not shrink while waiting
                st["wait"] += 1
                if st["wait"] == 1:
                    st["size0"] = size
                elif st["wait"] % STUCK_STEPS == 0:
                    st["blocked"] = size > 0.8 * st["size0"]
                    st["size0"] = size
            else:
                st["waited"] += st["wait"]
                st["wait"] = 0
                nxt = grip_at(path, seq[st["k"] + 1])
                st["k"] += 1
                if nxt != st["grip"]:
                    st["pending"], st["settle"] = nxt, 2 * SETTLE
                    st["grip_changes"] += 1
        w = path[seq[st["k"]]]
        playing = st["k"] != st["k_prev"] and not st["settle"]
        st["qt"], st["qd"], st["k_prev"] = w["q"], (-w["qd"] if playing else np.zeros(7)), st["k"]
        db = w["base"][:2] - now["base"][:2]              # base: to the entry, in the base frame
        c, s = np.cos(now["base"][2]), np.sin(now["base"][2])
        local = np.array([c * db[0] + s * db[1], -s * db[0] + c * db[1], float(_wrap(w["base"][2] - now["base"][2]))])
        cmd = np.zeros(3) if st["settle"] else np.clip(local / BASE_SPEED, -1, 1)
        torso = float(np.clip(TORSO_K * (w["torso"] - now["torso"]), -1, 1))
        np.copyto(action["action.base_motion"], np.array([cmd[0], cmd[1], cmd[2], torso]))
        np.copyto(action["action.gripper_close"], np.array([st["grip"]]))
        np.copyto(action["action.control_mode"], np.array([1.0]))
        action["action.end_effector_position"][:] = 0.0
        action["action.end_effector_rotation"][:] = 0.0

    def back():
        a, b, y = _gap(rec.pose(None), path[start])
        return a < BACK_TOL[0] and b < BACK_TOL[1] and y < BACK_TOL[2], (a, b, y)

    def at_rest():
        return float(np.abs(d.qvel[rec.arm_dof]).max()) < REST

    n0, errs, chunks, final_chunks = len(ep.chunks), [], 0, 0
    ctrl.run_controller = run_controller
    rec.override = override
    try:
        while chunks < max_chunks and not ep.done and not st["blocked"]:
            if st["final"]:                                   # arrived: hold the start entry until at rest
                if at_rest() or final_chunks >= FINAL_MAX:
                    break
                final_chunks += 1
            elif st["k"] >= len(seq) - 1 and not st["settle"]:
                ok, gap = back()
                errs.append(gap[0] + 10 * gap[1] + gap[2])
                if ok:
                    st["final"] = True
                    if at_rest():
                        break
                    final_chunks += 1
                elif len(errs) > 2 and errs[-3] - errs[-1] < 0.01:   # at the last entry, no closer over 2 chunks
                    break
            n = STEPS_PER_CHUNK
            ep.step({"action.end_effector_position": np.zeros((n, 3)), "action.end_effector_rotation": np.zeros((n, 3)),
                     "action.gripper_close": np.zeros((n, 1)), "action.base_motion": np.zeros((n, 4)),
                     "action.control_mode": np.ones((n, 1))})
            chunks += 1
            if on_chunk is not None:
                on_chunk(ep, st)
    finally:
        rec.override = None
        del ctrl.run_controller          # back to the OSC, its goal at the start entry's offset from the reached pose
        # (in the base frame, as the OSC keeps it; ref_pos / origin are from the last control substep; no forward(),
        # which would overwrite the solver's warm start, part of the integration state)
        ctrl.goal_pos = ctrl.world_to_origin_frame(np.array(ctrl.ref_pos))
        ctrl.goal_ori = ctrl.goal_origin_to_eef_pose()[:3, :3]
        g0 = path[start].get("goal")
        if g0 is not None and back()[0]:
            ctrl.goal_pos = ctrl.goal_pos + g0[0]
            ctrl.goal_ori = ctrl.goal_ori @ g0[1]
    ok, (qe, be, ye) = back()
    return dict(chunks=chunks, steps=sum(c["n"] for c in ep.chunks[n0:]), entries=len(seq),
                path_steps=len(path) - 1 - start, grip_changes=st["grip_changes"], arrived=bool(ok),
                blocked=st["blocked"], entry=int(seq[st["k"]]), waited=st["waited"] + st["wait"],
                final_chunks=final_chunks, joint_rad=round(qe, 3), base_cm=round(100 * be, 1), base_rad=round(ye, 3))
