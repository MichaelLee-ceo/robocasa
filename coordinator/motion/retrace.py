"""The robot's own trace, and the retrace that plays it back (the recovery that works, 2026-11-06).

Copied from runs/20261105_recovery/retrace.py (v2, which produced the results there: fixed 12 % -> 37 %, success
4 % -> 22 % against same-scene reruns); that copy stays as it was, this one is the one to develop.

RobotTrace(ep): wraps envkit's per-step log (changes nothing) and keeps, after every env step, the robot's measured
state: the 7 arm joints (encoders), the base pose (mobilebase0_base: world x, y, yaw; read from the simulator here,
on a real robot it is odometry), the hand pose in the arm's base frame, and the gripper command and control mode of
that step. Entry 0 is the state when it was attached. While `override` is set it is called before each env step with
that step's action (a view into the chunk, so changes go into the record too). This is the memory the retrace needs.

plan(path, start): the recorded entries backwards, from the last one to `start`. Loops are cut only where the robot
came back to practically the same state (every joint within 0.05 rad, base within 2 cm and 0.03 rad) with no gripper
change in between; entries that differ from the previous kept one by less than 0.01 rad / 0.5 cm (standing still)
are dropped. Otherwise the path is reversed as it was: no shortcuts.

retrace(ep, trace, start): plays the planned entries back, closed loop at every env step:
- arm: joint-space impedance toward the entry's joints (torque = M (KP e - KD qdot) + gravity and Coriolis, KP 150,
  critically damped; MuJoCo clamps it to the same actuator limits as the OSC), in place of the arm's OSC for the
  duration; the OSC goal is then set to the reached pose (in the base frame, as the OSC keeps it). Tracking joints,
  not the hand pose, keeps the redundant arm in the configuration it had;
- base: velocity commands in base mode toward the entry's base pose, from the measured pose at every step;
- gripper: the command in effect at that entry (so it opens where it had closed), holding still SETTLE steps before
  and after each change;
- speed: one recorded entry per step (the original speed), paused while the joints lag by more than TRACK or the
  base by more than two steps.
Nothing in the world is set: objects move only by contact; what fell stays. It ends when the robot is back (joints
within 0.03 rad, base within 1.5 cm and 0.02 rad of the start entry), when at the last entry it got no closer over 2
chunks, or after max_chunks. Its env steps are reported (`steps`) so the caller can leave them out of GR00T's time."""
import numpy as np

from ..schema import STEPS_PER_CHUNK

BASE_SPEED = np.array([0.030, 0.029, 0.066])   # base displacement per unit command per step: x, y (m), yaw (rad)
V_BASE, V_YAW = 0.03, 0.06                      # base step limits
KP_J = 150.0
KD_J = 2 * np.sqrt(KP_J)
SPEEDUP, TRACK = 1, 0.10                      # one recorded entry per step: the original speed
SAME = (0.05, 0.02, 0.03)                       # loop cut: joints rad (max), base m, base rad
STILL = (0.01, 0.005, 0.005)                    # dropped as standing still
SETTLE = 8
BACK_TOL = (0.03, 0.015, 0.02)


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RobotTrace:
    def __init__(self, ep):
        e = self.e = ep.kenv
        m = e.sim.model
        ctrl = self.ctrl = e.robots[0].part_controllers["right"]
        self.origin = m.site_name2id(f"{ctrl.naming_prefix}{ctrl.part_name}_center")
        self.hand = e.robots[0].eef_site_id["right"]
        self.base = m.body_name2id("mobilebase0_base")
        self.arm = np.array([m.jnt_qposadr[m.joint_name2id(f"robot0_joint{i}")] for i in range(1, 8)])
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
        o, Ro = d.site_xpos[self.origin], d.site_xmat[self.origin].reshape(3, 3)
        p, Rp = d.site_xpos[self.hand], d.site_xmat[self.hand].reshape(3, 3)
        b, Rb = d.xpos[self.base], d.xmat[self.base].reshape(3, 3)
        g = md = None
        if action is not None:
            g = float(np.asarray(action["action.gripper_close"]).ravel()[0])
            md = float(np.asarray(action["action.control_mode"]).ravel()[0])
        return dict(q=d.qpos[self.arm].copy(), base=np.array([b[0], b[1], _yaw(Rb)]), rel=Ro.T @ (p - o),
                    R=Ro.T @ Rp, grip=g, mode=md)


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
    """absolute indices of the entries to play back, from the last one to `start`"""
    n = len(path)
    g = [grip_at(path, i) for i in range(n)]
    changes = np.cumsum([0] + [g[i] != g[i - 1] for i in range(1, n)])
    Q = np.array([p["q"] for p in path])
    B = np.array([p["base"] for p in path])
    rev, cur = [n - 1], n - 1
    while cur > start:
        js = np.arange(start, cur - 1)
        if len(js):
            js = js[changes[js] == changes[cur]]
        if len(js):
            ok = (np.abs(Q[js] - Q[cur]).max(axis=1) < SAME[0]) & \
                 (np.linalg.norm(B[js, :2] - B[cur, :2], axis=1) < SAME[1]) & \
                 (np.abs(_wrap(B[js, 2] - B[cur, 2])) < SAME[2])
            cur = int(js[ok].min()) if ok.any() else cur - 1
        else:
            cur -= 1
        rev.append(cur)
    out = [rev[0]]
    for i in rev[1:]:
        q, b, y = _gap(path[i], path[out[-1]])
        if g[i] != g[out[-1]] or i == start or q >= STILL[0] or b >= STILL[1] or y >= STILL[2]:
            out.append(i)
    return out


def retrace(ep, rec, start, max_chunks=400, on_chunk=None):
    """play the recorded path of RobotTrace `rec` back to entry `start` (see the module doc). Returns a summary dict."""
    path = list(rec.path)
    seq = plan(path, start)
    ctrl, e = rec.ctrl, rec.e
    st = dict(k=0, settle=0, pending=None, grip=grip_at(path, len(path) - 1), qt=path[seq[0]]["q"].copy(),
              steps=0, grip_changes=0)

    def run_controller():  # joint-space impedance toward st["qt"], in place of the OSC
        ctrl.update()
        acc = KP_J * (st["qt"] - ctrl.joint_pos) - KD_J * ctrl.joint_vel
        ctrl.torques = ctrl.mass_matrix @ acc + ctrl.torque_compensation
        ctrl.new_update = True
        return ctrl.torques

    def override(action):
        now = rec.pose(None)
        if st["settle"]:
            st["settle"] -= 1
            if st["settle"] == SETTLE and st["pending"] is not None:
                st["grip"], st["pending"] = st["pending"], None
        else:
            for _ in range(SPEEDUP):  # advance while the arm keeps up
                cur_w = path[seq[st["k"]]]
                if st["k"] + 1 >= len(seq) or np.abs(now["q"] - cur_w["q"]).max() > TRACK \
                        or np.linalg.norm(now["base"][:2] - cur_w["base"][:2]) > 2 * V_BASE \
                        or abs(_wrap(now["base"][2] - cur_w["base"][2])) > 2 * V_YAW:
                    break
                nxt = grip_at(path, seq[st["k"] + 1])
                st["k"] += 1
                if nxt != st["grip"]:
                    st["pending"], st["settle"] = nxt, 2 * SETTLE
                    st["grip_changes"] += 1
                    break
        w = path[seq[st["k"]]]
        st["qt"] = w["q"]
        db = w["base"][:2] - now["base"][:2]
        nb = np.linalg.norm(db)
        sb = db if nb <= V_BASE else db * V_BASE / nb
        dy = float(np.clip(_wrap(w["base"][2] - now["base"][2]), -V_YAW, V_YAW))
        c, s = np.cos(now["base"][2]), np.sin(now["base"][2])
        local = np.array([c * sb[0] + s * sb[1], -s * sb[0] + c * sb[1]])
        cmd = np.clip(np.array([local[0], local[1], dy]) / BASE_SPEED, -1, 1)
        if st["settle"]:
            cmd[:] = 0.0
        np.copyto(action["action.base_motion"], np.array([cmd[0], cmd[1], cmd[2], 0.0]))
        np.copyto(action["action.gripper_close"], np.array([st["grip"]]))
        np.copyto(action["action.control_mode"], np.array([1.0]))
        action["action.end_effector_position"][:] = 0.0
        action["action.end_effector_rotation"][:] = 0.0
        st["steps"] += 1

    def back():
        a, b, y = _gap(rec.pose(None), path[start])
        return a < BACK_TOL[0] and b < BACK_TOL[1] and y < BACK_TOL[2], (a, b, y)

    n0, errs, chunks = len(ep.chunks), [], 0
    ctrl.run_controller = run_controller
    rec.override = override
    try:
        while chunks < max_chunks and not ep.done:
            if st["k"] >= len(seq) - 1 and not st["settle"]:
                ok, gap = back()
                errs.append(gap[0] + 10 * gap[1] + gap[2])
                if ok or (len(errs) > 2 and errs[-3] - errs[-1] < 0.01):
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
        del ctrl.run_controller          # back to the OSC; its goal = the reached pose, in the base frame as the OSC
        # keeps it (ref_pos / origin are from the last control substep; no forward(), which would overwrite the
        # solver's warm start, part of the integration state)
        ctrl.goal_pos = ctrl.world_to_origin_frame(np.array(ctrl.ref_pos))
        ctrl.goal_ori = ctrl.goal_origin_to_eef_pose()[:3, :3]
    ok, (qe, be, ye) = back()
    return dict(chunks=chunks, steps=sum(c["n"] for c in ep.chunks[n0:]), entries=len(seq), path_steps=len(path) - 1 - start,
                grip_changes=st["grip_changes"], arrived=bool(ok), joint_rad=round(qe, 3), base_cm=round(100 * be, 1),
                base_rad=round(ye, 3))
