"""Does the retrace follow the recorded path, and is the robot in the same state when GR00T starts again?

For each phase 0 failure point: replay the plain run to the point exactly as loop.py does (each stage's entry kept as
loop.py's remember() keeps it), saving the full robot state at the failed stage's entry; then the first retrace back to
it (the same for every policy seed: the path is recorded and the retrace is deterministic), logged at every env step.

path      at every retrace step, its distance to the recorded path (the nearest of the planned entries, separately in
          each space): arm joints (rad, max over the 7), hand position in the world (cm), base (cm), torso (cm); and at
          chunk ends the lag behind the entry being tracked
restart   after the retrace, against the stage's entry: the robot's joints (arm, fingers, base, torso) and their speeds,
          the hand pose, the gripper command and the gripper's command ramp (PandaGripper.current_action), the OSC
          goal's offset from the hand, GR00T's proprio inputs (the obs state.* fields), and every task object (not
          restored by design: only what the robot carried comes back with it)
objects   every task object, by what happened to it in the stage (robocasa's check_obj_grasped at every step):
          carried (grasped in the stage and still held at the failure point: the retrace should carry it back and put
          it down where it was picked up), dropped (grasped, then lost before the failure point: it stays), pushed
          (moved 2 cm or more without being grasped: it stays), still; for each, where it ends against where it was
          at the stage start and where it was picked up

usage: python -I coordinator/tests/check_retrace_state.py <point id> ...   (one JSON line per point)
Results of the versions compared on 2026-11-07 (v2, v3, C = the final one): runs/20261106_coordinator/README.md."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from coordinator import paths  # noqa: E402
from coordinator.motion import retrace as rt  # noqa: E402
from coordinator.points import load_point  # noqa: E402

paths.use_envkit()
import envkit  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402


def angle(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def robot_state(ep, trace):
    e, m, d = ep.kenv, ep.kenv.sim.model, ep.kenv.sim.data
    ctrl = trace.ctrl
    rob = [j for j in range(m.njnt) if (m.joint_id2name(j) or "").startswith(("robot0_", "mobilebase0_", "gripper0_"))]
    grip = e.robots[0].gripper["right"]
    acts = [a for a in range(m.nu) if (m.actuator_id2name(a) or "").startswith("gripper0")]
    fing = [j for j in rob if m.joint_id2name(j).startswith("gripper0")]
    arm_act = [m.actuator_name2id(f"robot0_torq_j{i}") for i in range(1, 8)]
    torso_ctrl = e.robots[0].composite_controller.part_controllers.get("torso")
    goal_ori_now = ctrl.goal_origin_to_eef_pose()[:3, :3]
    return dict(
        qpos={m.joint_id2name(j): d.qpos[m.jnt_qposadr[j]].copy() for j in rob},
        qvel={m.joint_id2name(j): d.qvel[m.jnt_dofadr[j]].copy() for j in rob},
        fingers=np.array([d.qpos[m.jnt_qposadr[j]] for j in fing]),
        base=trace.pose(None)["base"], q=trace.pose(None)["q"],
        hand_p=d.site_xpos[trace.hand].copy(), hand_R=d.site_xmat[trace.hand].reshape(3, 3).copy(),
        ramp=np.array(grip.current_action, dtype=float).ravel().copy(), grip_ctrl=d.ctrl[acts].copy(),
        osc_goal_cm=100 * float(np.linalg.norm(np.asarray(ctrl.goal_pos) - ctrl.world_to_origin_frame(np.array(ctrl.ref_pos)))),
        osc_goal_deg=angle(np.asarray(ctrl.goal_ori), goal_ori_now),
        torso_goal=None if torso_ctrl is None or torso_ctrl.goal_qpos is None
        else float(np.asarray(torso_ctrl.goal_qpos).ravel()[0]),
        arm_torque=d.ctrl[arm_act].copy(),
        obs={k: np.asarray(v, dtype=float).copy() for k, v in ep.obs.items() if k.startswith("state.")},
        objects={o: (d.body_xpos[e.obj_body_id[o]].copy(), d.body_xmat[e.obj_body_id[o]].reshape(3, 3).copy())
                 for o in e.objects})


def compare(a, b):
    mx = lambda x: round(float(np.max(np.abs(x))) if np.size(x) else 0.0, 4)  # noqa: E731
    arm = [k for k in a["qpos"] if k.startswith("robot0_joint")]
    other = [k for k in a["qpos"] if k not in arm and not k.startswith("gripper0")]
    return dict(
        arm_rad=mx(a["q"] - b["q"]),
        arm_speed_before=mx([a["qvel"][k] for k in arm]), arm_speed_after=mx([b["qvel"][k] for k in arm]),
        base_cm=round(100 * float(np.linalg.norm(a["base"][:2] - b["base"][:2])), 2),
        base_yaw_rad=round(abs(float(rt._wrap(a["base"][2] - b["base"][2]))), 4),
        other_joints={k: mx(a["qpos"][k] - b["qpos"][k]) for k in other},
        hand_mm=round(1000 * float(np.linalg.norm(a["hand_p"] - b["hand_p"])), 2),
        hand_deg=round(angle(a["hand_R"], b["hand_R"]), 3),
        fingers_mm=round(1000 * mx(a["fingers"] - b["fingers"]), 2),
        ramp=(a["ramp"].tolist(), b["ramp"].tolist()), grip_ctrl=(a["grip_ctrl"].round(3).tolist(), b["grip_ctrl"].round(3).tolist()),
        osc_goal_offset_cm=(round(a["osc_goal_cm"], 2), round(b["osc_goal_cm"], 2)),
        osc_goal_offset_deg=(round(a["osc_goal_deg"], 2), round(b["osc_goal_deg"], 2)),
        torso_goal_cm=None if a["torso_goal"] is None or b["torso_goal"] is None else round(100 * abs(a["torso_goal"] - b["torso_goal"]), 3),
        arm_torque=mx(a["arm_torque"] - b["arm_torque"]),
        gr00t_proprio={k: mx(a["obs"][k] - b["obs"][k]) for k in a["obs"]},
        objects_cm={o: round(100 * float(np.linalg.norm(a["objects"][o][0] - b["objects"][o][0])), 1) for o in a["objects"]},
        objects_deg={o: round(angle(a["objects"][o][1], b["objects"][o][1]), 1) for o in a["objects"]})


def check(pid):
    point, rec = load_point(pid)
    ep = envkit.Episode(point["task"], point["seed"], horizon=10 ** 6)
    d = ep.kenv.sim.data
    trace = rt.RobotTrace(ep)
    hand = [d.site_xpos[trace.hand].copy()]
    torso = [d.qpos[trace.torso]]
    phase, log = ["forward"], []
    objs = list(ep.kenv.objects)
    held = {o: [] for o in objs}              # (entry, grasped, position) at every step of the stage and the retrace
    inner = ep._log.step

    def step(action):
        out = inner(action)
        hand.append(d.site_xpos[trace.hand].copy())
        torso.append(float(d.qpos[trace.torso]))
        if phase[0] == "back":
            log.append(len(trace.path) - 1)
        if phase[0] in ("stage", "back"):
            for o in objs:
                held[o].append((phase[0], len(trace.path) - 1, bool(OU.check_obj_grasped(ep.kenv, o)),
                                d.body_xpos[ep.kenv.obj_body_id[o]].copy()))
        return out
    ep._log.step = step

    starts, reached, S0 = {}, -1, None
    for i, c in enumerate(rec["chunks"][:point["chunk"] + 1]):
        while reached < min(rec["tracks"][i]["stage"], point["stage"]):
            reached += 1
            starts[reached] = len(trace.path) - 1
            if reached == point["stage"]:
                S0 = robot_state(ep, trace)
                phase[0] = "stage"
        ep.step(c["actions"])
    start = starts[point["stage"]]
    at_fail = robot_state(ep, trace)
    path = list(trace.path)
    seq = rt.plan(path, start)
    Q = np.array([path[i]["q"] for i in seq])
    B = np.array([path[i]["base"][:2] for i in seq])
    H = np.array([hand[i] for i in seq])
    T = np.array([torso[i] for i in seq])
    phase[0] = "back"
    lag = []

    def on_chunk(ep, st):
        lag.append(float(np.max(np.abs(trace.pose(None)["q"] - path[seq[st["k"]]]["q"]))))

    res = rt.retrace(ep, trace, start, 400, on_chunk=on_chunk)
    stalled = bool(res["blocked"])
    S1 = robot_state(ep, trace)
    objects = {}
    for o in objs:
        h = held[o]
        st_ = [x for x in h if x[0] == "stage"]
        bk = [x for x in h if x[0] == "back"]
        p0, p1 = S0["objects"][o][0], S1["objects"][o][0]
        pick = next((x for x in st_ if x[2]), None)
        held_end = bool(st_ and st_[-1][2])
        moved = float(np.linalg.norm(at_fail["objects"][o][0] - p0))
        kind = "carried" if pick and held_end else "dropped" if pick else "pushed" if moved >= 0.02 else "still"
        release = next((x for x in reversed(bk) if x[2]), None)
        objects[o] = dict(kind=kind, moved_in_stage_cm=round(100 * moved, 1),
                          restart_vs_start_cm=round(100 * float(np.linalg.norm(p1 - p0)), 1),
                          restart_vs_pick_cm=None if pick is None else round(100 * float(np.linalg.norm(p1 - pick[3])), 1),
                          carried_back=any(x[2] for x in bk), pick_entry=None if pick is None else pick[1],
                          release_entry=None if release is None else release[1])
    # distance to the recorded path, every retrace step (nearest planned entry, in each space separately)
    dev_q, dev_h, dev_b, dev_t = [], [], [], []
    for idx in log:
        dev_q.append(float(np.abs(Q - trace.path[idx]["q"]).max(axis=1).min()))
        dev_h.append(100 * float(np.linalg.norm(H - hand[idx], axis=1).min()))
        dev_b.append(100 * float(np.linalg.norm(B - trace.path[idx]["base"][:2], axis=1).min()))
        dev_t.append(100 * float(np.abs(T - torso[idx]).min()))
    pct = lambda x, q: round(float(np.percentile(x, q)), 3) if len(x) else None  # noqa: E731
    out = dict(id=pid, task=point["task"], stage=point["stage"], path_steps=len(path) - 1 - start, entries=len(seq),
               retrace=res, stalled=stalled, retrace_steps=len(log),
               path=dict(joints_rad=dict(median=pct(dev_q, 50), p95=pct(dev_q, 95), max=pct(dev_q, 100)),
                         hand_cm=dict(median=pct(dev_h, 50), p95=pct(dev_h, 95), max=pct(dev_h, 100)),
                         base_cm=dict(median=pct(dev_b, 50), p95=pct(dev_b, 95), max=pct(dev_b, 100)),
                         torso_cm=dict(median=pct(dev_t, 50), p95=pct(dev_t, 95), max=pct(dev_t, 100)),
                         lag_rad=dict(median=pct(lag, 50), max=pct(lag, 100))),
               restart=compare(S0, S1),
               objects_moved_in_stage_cm={o: round(100 * float(np.linalg.norm(S0["objects"][o][0] - at_fail["objects"][o][0])), 1)
                                          for o in S0["objects"]},
               objects=objects)
    ep.close()
    return out


if __name__ == "__main__":
    for pid in sys.argv[1:]:
        try:
            print(json.dumps(check(pid), default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)), flush=True)
        except Exception as x:
            import traceback
            print(json.dumps(dict(id=pid, error=f"{type(x).__name__}: {x}", trace=traceback.format_exc()[-1500:])), flush=True)
