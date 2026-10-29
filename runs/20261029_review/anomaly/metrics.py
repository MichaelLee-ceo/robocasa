"""Physics-anomaly scan, step 2 (library): per-rollout metrics and flags from feat/<tag>.pkl (scan.py).

Every rollout's first 32 steps are compared with the main run's next 32 steps (ep_states[t : t + 33]); n = the number of
states both have (a rollout that finished its stage at step 16 has 17).
Flags (thresholds as in the review brief):
  F1a object moved > 10 cm while it moved < 2 cm in the reference
  F1b object fell > 10 cm (absolute: start height - lowest height)
  F1r object fell > 10 cm while it fell < 5 cm in the reference
  F1c object faster than 2 m/s (finite difference of its position over one 0.05 s step)
  F2  base displacement > 5 cm in both and the angle between them > 90 deg
  F3a end effector faster than 1 m/s relative to the base
  F3b end-effector jump: one step's displacement relative to the base > 2 cm and > 4x the median step of that trajectory
  F4  fingers open > 5 mm within 8 steps while touching an object (window starts every 4 steps, s0 <= 24), the reference
      does not; F4e: the same at s0 = 0 only (the restore signature of bug 4)
  F5  joint-velocity spike: an arm joint > ARM_QV rad/s, a finger > FING_QV m/s, a fixture joint > FIX_QV, or an object
      angular velocity > OBJ_W rad/s (thresholds from the main runs' own distribution, see analyze.py)
The reference is the main run (vs="main") or, as the null, a sibling candidate's m0 rollout of the same point
(vs="sib": candidate (c + 1) mod N): the main run is one more draw of the same policy from the same observation, so with
a faithful restore both give the same rates."""
import glob, os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.05
ARM_QV, FING_QV, FIX_QV, OBJ_W = 3.0, 0.5, 20.0, 50.0


def load(tag=None):
    fs = sorted(glob.glob(os.path.join(HERE, "feat", f"{tag or '*'}.pkl")))
    for f in fs:
        with open(f, "rb") as fh:
            yield pickle.load(fh)


def window(main, t, n=33):
    """the main run's states [t, t + n) as a trajectory dict like a rollout's."""
    T = main["T"]
    e = min(T, t + n)
    w = {k: main[k][t:e] for k in ("time", "eef_w", "eef_b", "base", "gap", "obj", "qv_rob", "qv_obj", "qv_fix_max")}
    w["T"] = e - t
    ts = main["touch_steps"]
    sel = (ts >= t) & (ts < e)
    w["touch"], w["touch_steps"] = main["touch"][sel], ts[sel] - t
    w["trace_touch"] = main["trace_touch"][t:e]
    return w


def unwrap_dyaw(a, b):
    return float((b - a + np.pi) % (2 * np.pi) - np.pi)


def finger_events(tr, n, use_trace=False):
    """window starts s0 (every 4 steps, s0 <= 24) where the fingers touch an object and open > 5 mm within 8 steps.
    Returns (list of (s0, opening m), whether any touch at s0 = 0)."""
    ev = []
    if use_trace:
        starts = [s for s in range(0, min(25, n - 1), 4)]
        touch = {s: bool(tr["trace_touch"][s].any()) for s in starts}
    else:
        touch = {int(s): bool(tr["touch"][i].any()) for i, s in enumerate(tr["touch_steps"]) if s <= 24 and s < n - 1}
    for s0, tch in touch.items():
        if not tch:
            continue
        op = float(np.max(tr["gap"][s0 + 1:min(s0 + 9, n)]) - tr["gap"][s0])
        if op > 0.005:
            ev.append((s0, op))
    return ev, touch.get(0, False)


def traj_stats(tr, n):
    """absolute per-trajectory statistics over its first n states."""
    o = tr["obj"][:n]
    disp = np.linalg.norm(o - o[:1], axis=2).max(0)  # per object
    drop = o[0, :, 2] - o[:, :, 2].min(0)
    v_obj = (np.linalg.norm(np.diff(o, axis=0), axis=2).max(0) / DT) if n > 1 else np.zeros(o.shape[1])
    eb = tr["eef_b"][:n]
    steps = np.linalg.norm(np.diff(eb, axis=0), axis=1) if n > 1 else np.zeros(1)
    med = float(np.median(steps)) if len(steps) else 0.0
    qr = np.abs(tr["qv_rob"][:n])
    qo = tr["qv_obj"][:n]
    return dict(disp=disp, drop=drop, v_obj=v_obj, eef_v=float(steps.max() / DT), eef_step=steps,
                eef_jump=bool(np.any((steps > 0.02) & (steps > 4 * max(med, 1e-4)))),
                base_d=tr["base"][n - 1, :2] - tr["base"][0, :2], dyaw=unwrap_dyaw(tr["base"][0, 2], tr["base"][n - 1, 2]),
                arm_qv=float(qr[:, 4:11].max()), fing_qv=float(qr[:, 11:13].max()), base_qv=float(qr[:, 0:3].max()),
                obj_w=float(np.linalg.norm(qo[:, :, 3:], axis=2).max()) if qo.shape[1] else 0.0,
                obj_vq=float(np.linalg.norm(qo[:, :, :3], axis=2).max()) if qo.shape[1] else 0.0,
                fix_qv=float(tr["qv_fix_max"][:n].max()),
                gap8=float(tr["gap"][min(8, n - 1)] - tr["gap"][0]))


def flags(r, ref, n, ref_trace=False):
    """flags of trajectory r against reference ref (both truncated to n states)."""
    a, b = traj_stats(r, n), traj_stats(ref, n)
    f = {}
    f["F1a"] = bool(np.any((a["disp"] > 0.10) & (b["disp"] < 0.02)))
    f["F1b"] = bool(np.any(a["drop"] > 0.10))
    f["F1r"] = bool(np.any((a["drop"] > 0.10) & (b["drop"] < 0.05)))
    f["F1c"] = bool(np.any(a["v_obj"] > 2.0))
    na, nb = np.linalg.norm(a["base_d"]), np.linalg.norm(b["base_d"])
    ang = float(np.degrees(np.arccos(np.clip(a["base_d"] @ b["base_d"] / (na * nb), -1, 1)))) if na > 0.05 and nb > 0.05 else None
    f["F2"] = bool(ang is not None and ang > 90)
    f["F3a"] = a["eef_v"] > 1.0
    f["F3b"] = a["eef_jump"]
    ea, _ = finger_events(r, n)
    eb, _ = finger_events(ref, n, use_trace=ref_trace)
    f["F4"] = bool(ea) and not eb
    f["F4e"] = any(s == 0 for s, _ in ea) and not any(s == 0 for s, _ in eb)
    f["F5"] = a["arm_qv"] > ARM_QV or a["fing_qv"] > FING_QV or a["fix_qv"] > FIX_QV or a["obj_w"] > OBJ_W
    det = dict(base_ang=ang, base_d=float(na), base_d_ref=float(nb), dyaw=a["dyaw"], dyaw_ref=b["dyaw"],
               eef_v=a["eef_v"], arm_qv=a["arm_qv"], fing_qv=a["fing_qv"], fix_qv=a["fix_qv"], obj_w=a["obj_w"],
               obj_v=float(a["v_obj"].max()) if len(a["v_obj"]) else 0.0, gap8=a["gap8"], gap8_ref=b["gap8"],
               fing_ev=ea, fing_ev_ref=eb,
               obj_move=[int(i) for i in np.where((a["disp"] > 0.10) & (b["disp"] < 0.02))[0]],
               obj_fall=[int(i) for i in np.where(a["drop"] > 0.10)[0]], drop=a["drop"], drop_ref=b["drop"],
               disp=a["disp"], disp_ref=b["disp"])
    return f, det
