"""The end effector's commanded path over a candidate chunk, from the policy's action and observation alone (no
simulator), for a VLM critic that may not use the simulator (2026-10-30).

The arm controller (OSC_POSE, robosuite default_pandaomron.json) takes per step a delta in the robot base frame,
clipped to [-1, 1] and scaled to 0.05 m and 0.5 rad. The arm does not reach each step's target within the step: the path
follows the controller's critically damped response to the targets (track()), starting from the end effector pose in
the policy's request (state.end_effector_*_relative, base frame). The base frame is placed with the request's base pose
(state.base_position, state.base_rotation). Steps in base mode (control_mode > 0.5) move the base, not the arm: about
BASE_XY x (a[0], a[1]) in its current frame and BASE_YAW x a[2] (calibrated on the saved base joints). The gripper is
commanded closed where gripper_close > 0.5. model="ideal" (each target reached within the step) is kept for comparison.
validate(): compares the commanded path with the path the simulator executed (from the saved states, forward kinematics
only), to measure how faithful it is.
usage: python -I commanded.py validate [n_points]"""
import glob, json, os, sys
import numpy as np

R = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # runs/
sys.path[:0] = [f"{R}/scripts", f"{R}/20261028_critical", f"{R}/20261025_seen480"]
STEP_POS, STEP_ROT = 0.05, 0.5
MODEL = os.environ.get("CMD_MODEL", "osc")


def quat2mat(q):
    """robosuite order x y z w"""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axisangle2mat(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


KP, SUB, DT = 150.0, 25, 0.002  # OSC stiffness (critically damped), physics substeps per control step, substep length
# base-mode steps (calibrated 2026-10-30 on 7,964 base-mode steps of SteamInMicrowave / PackIdenticalLunches rollouts,
# from the base joints in the saved states): per step the base moves BASE_XY * (a[0], a[1]) in its current frame
# (R^2 0.82-0.90) and turns BASE_YAW * a[2] (R^2 0.96-0.99); the torso does not move. The arm holds its pose in the base
# frame meanwhile (its goal is kept in base mode).
BASE_XY, BASE_YAW = 0.028, 0.058


def track(targets_rel, w=np.sqrt(KP), v0=None):
    """position response of a critically damped second-order tracker: at each control step the target is the current
    position plus targets_rel[i]; the target is held for SUB substeps of DT; velocity carries over. (n, d) -> (n + 1, d)
    displacements from the start."""
    d = targets_rel.shape[1]
    x, v, out = np.zeros(d), np.zeros(d) if v0 is None else np.asarray(v0, float), [np.zeros(d)]
    for rel in targets_rel:
        goal = x + rel
        for _ in range(SUB):
            acc = w * w * (goal - x) - 2 * w * v
            v = v + acc * DT
            x = x + v * DT
        out.append(x.copy())
    return np.array(out)


def commanded(snap, chunk, model="osc"):
    """snap: the decision point's snapshot (npz); chunk: {action key: (16, d)} of one candidate.
    model "osc": the arm follows each step's target as the controller's critically damped response (track()); "ideal": it
    reaches every target within the step. Returns world positions (17, 3), world rotations (17, 3, 3), gripper commanded
    closed (16,), base mode (16,)."""
    bp = np.asarray(snap["req/state.base_position"]).reshape(-1)[:3]
    Rb = quat2mat(np.asarray(snap["req/state.base_rotation"]).reshape(-1)[:4])
    p = np.asarray(snap["req/state.end_effector_position_relative"], float).reshape(-1)[:3].copy()
    Rm = quat2mat(np.asarray(snap["req/state.end_effector_rotation_relative"]).reshape(-1)[:4])
    a_pos = np.clip(np.asarray(chunk["action.end_effector_position"], float), -1, 1)
    a_rot = np.clip(np.asarray(chunk["action.end_effector_rotation"], float), -1, 1)
    base_mode = np.asarray(chunk["action.control_mode"], float).reshape(-1) > 0.5
    arm = (~base_mode)[:, None]
    a_base = np.asarray(chunk["action.base_motion"], float)
    # base pose per step (world): moves only in base-mode steps
    bps, Rbs = [bp.copy()], [Rb.copy()]
    for i in range(len(base_mode)):
        b, R_ = bps[-1], Rbs[-1]
        if base_mode[i]:
            b = b + R_ @ np.array([BASE_XY * a_base[i, 0], BASE_XY * a_base[i, 1], 0.0])
            R_ = R_ @ axisangle2mat(np.array([0.0, 0.0, BASE_YAW * a_base[i, 2]]))
        bps.append(b); Rbs.append(R_)
    bps, Rbs = np.array(bps), np.array(Rbs)
    if model == "osc":
        P = p + track(STEP_POS * a_pos * arm)
        ang = track(STEP_ROT * a_rot * arm)  # small-angle: the rotation increments follow the same response
        Rs = [Rm.copy()]
        for i in range(len(a_rot)):
            Rs.append(axisangle2mat(ang[i + 1] - ang[i]) @ Rs[-1])
        Rs = np.array(Rs)
    else:
        P, Rs = [p.copy()], [Rm.copy()]
        for i in range(len(a_pos)):
            if not base_mode[i]:
                p = p + STEP_POS * a_pos[i]
                Rm = axisangle2mat(STEP_ROT * a_rot[i]) @ Rm
            P.append(p.copy()); Rs.append(Rm.copy())
        P, Rs = np.array(P), np.array(Rs)
    return dict(pos=bps + np.einsum("nij,nj->ni", Rbs, P), rot=np.einsum("nij,njk->nik", Rbs, Rs), base=bps,
                close=np.asarray(chunk["action.gripper_close"], float).reshape(-1) > 0.5, base_mode=base_mode)


def first_chunk(npz):
    z = np.load(npz)
    return {k: z[k][0] for k in z.files if k.startswith("action.")}, z["states"][:17]


def validate(n_points=12):
    """commanded vs executed end effector path (forward kinematics of the saved states) at steps 4, 8 and 16"""
    import data
    from analyze_stages import replay_env
    from oracle import reset_to, set_state
    out, envs = [], {}
    for task in data.TASKS:
        rs = list(data.rows(task, "confirm").values())[:n_points // 3]
        for r in rs:
            if task not in envs:
                envs[task] = replay_env(task)
            env = envs[task]
            z = np.load(data.scenes()[r["tag"]], allow_pickle=True)
            reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
            site = env.robots[0].eef_site_id["right"]
            s = np.load(data.snap(r))
            for i in range(8):
                chunk, states = first_chunk(f"{data.roll(r)}/{r['tag']}_t{r['t']}_c{i}_m0.npz")
                c = commanded(s, chunk, MODEL)
                ex = []
                for f in range(17):
                    set_state(env, states[f]); ex.append(np.array(env.sim.data.site_xpos[site]))
                ex = np.array(ex)
                err = np.linalg.norm(c["pos"] - ex, axis=1)
                out.append(dict(task=task, tag=r["tag"], t=r["t"], cand=i, start_err_cm=round(100 * err[0], 2),
                                err_cm={s_: round(100 * err[s_], 2) for s_ in (4, 8, 16)},
                                moved_cm=round(100 * float(np.linalg.norm(ex[16] - ex[0])), 1), base_steps=int(c["base_mode"].sum())))
    for task in data.TASKS:
        o = [x for x in out if x["task"] == task]
        if o:
            e16 = np.array([x["err_cm"][16] for x in o]); mv = np.array([x["moved_cm"] for x in o])
            print(f"{task}: {len(o)} candidates, start error max {max(x['start_err_cm'] for x in o):.2f} cm, end error median "
                  f"{np.median(e16):.2f} cm (90 % {np.percentile(e16, 90):.2f}), executed motion median {np.median(mv):.1f} cm, "
                  f"candidates with base-mode steps {sum(x['base_steps'] > 0 for x in o)}")
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["validate"]:
        res = validate(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
        json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate.json"), "w"), indent=0)
