"""Per-point integrity scan of the collected rollouts (no simulation; reads only).

For every error-free row of every dir in common.py (current, superseded, dense, confirm, the _bug3 positive controls,
dense2, and a sample of the bug-1 dirs as a positive control), loads every roll/<tag>_t<t>_c<i>_m<m>.npz fully (each
zip member is read whole, so its CRC is verified), the main run's states, and the snap's state, and records:
  - file health: missing / unreadable npz, mp4 (exists, non-empty, has a moov atom), snap (exists, CRC of every member)
  - row vs npz: len(states) vs steps, chunks vs steps, action shapes, time column (0.05 s per step), finite values
  - start state: states[0] vs the main run's states[t] (bitwise), snap state vs states[t], obs_fp per rollout
  - shared first chunk (per candidate): chunk-0 arrays and states[0..16] bitwise across continuations, first row
    where two continuations differ, whether chunk 1 differs
  - seeds: pairs of candidates with bitwise-equal chunk 0
  - sha1 of chunk 0 / chunk 1 / states[0..16] per rollout (cross-dir comparisons are done in report.py)
  - physical signatures of the known restore bugs, from the state arrays and the scene's joint layout (layout.json):
    finger gap over the first 16 steps and the commanded gripper side (bug 4), arm joint speed over the first 16 steps
    (bug 2), the direction of the realized base velocity vs the commanded one under the correct and the bug-3
    transform (bug 3), the height of every free object at a few steps (drops).
Writes scan.jsonl (one record per point). usage: python -I scan.py [workers=3] [--only DIR ...]"""
import glob, hashlib, io, json, os, sys, time, traceback, zipfile
from multiprocessing import get_context
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, SEEN, HERE, COARSE, OTHER, coarse_dirs, read_rows, scenes  # noqa: E402

KEYS = ["action.end_effector_position", "action.end_effector_rotation", "action.gripper_close", "action.base_motion",
        "action.control_mode"]
DIMS = {"action.end_effector_position": 3, "action.end_effector_rotation": 3, "action.gripper_close": 1,
        "action.base_motion": 4, "action.control_mode": 1}
OLD_OPEN = abs(0.020833 - -0.020833) * 1.9  # the pre-v4 settle rule: fully open = 1.9 x the init gap
_EP = {}


def h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def load_full(path):
    """every member read whole (zipfile checks its CRC), parsed as .npy"""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            data = zf.read(info.filename)
            out[info.filename[:-4] if info.filename.endswith(".npy") else info.filename] = \
                np.lib.format.read_array(io.BytesIO(data), allow_pickle=False)
    return out


def mp4_ok(path):
    if not os.path.exists(path):
        return "missing"
    sz = os.path.getsize(path)
    if sz == 0:
        return "empty"
    with open(path, "rb") as f:
        f.seek(max(0, sz - 262144)); tail = f.read()
        if b"moov" in tail:
            return "ok"
        f.seek(0)
        return "ok" if b"moov" in f.read() else "no_moov"


def ep_states(task, tag):
    key = (task, tag)
    if key not in _EP:
        if len(_EP) > 6:
            _EP.clear()
        _EP[key] = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
    return _EP[key]


def idx(lay):
    J = lay["joints"]; nq = lay["nq"]
    q = lambda n: 1 + J[n]["qpos"]
    v = lambda n: 1 + nq + J[n]["dof"]
    return dict(
        fwd_q=q("mobilebase0_joint_mobile_forward"), side_q=q("mobilebase0_joint_mobile_side"),
        yaw_q=q("mobilebase0_joint_mobile_yaw"), fwd_v=v("mobilebase0_joint_mobile_forward"),
        side_v=v("mobilebase0_joint_mobile_side"),
        fwd_rev=J["mobilebase0_joint_mobile_forward"]["axis"] == [0.0, 1.0, 0.0],
        arm_v=[v(f"robot0_joint{i}") for i in range(1, 8)],
        f1=q("gripper0_right_finger_joint1"), f2=q("gripper0_right_finger_joint2"),
        open_gap=max(abs(J["gripper0_right_finger_joint1"]["range"][1] - J["gripper0_right_finger_joint2"]["range"][0]),
                     abs(J["gripper0_right_finger_joint1"]["range"][0] - J["gripper0_right_finger_joint2"]["range"][1])),
        objs={n[:-len("_joint0")]: 1 + J[n]["qpos"] for n in J if J[n]["type"] == 0 and n.endswith("_joint0")},
        state_len=lay["state_len"])


def base_dir_test(st, bm, ix, yaw0_main):
    """median |angle| between the realized slide-joint velocity after step g and the commanded base velocity of step g
    rotated as the controller does, with the start pose of the episode (correct) or of the rollout (bug 3)"""
    n = min(len(st) - 1, bm.shape[0])
    if n <= 0:
        return None
    yaw = st[:n, ix["yaw_q"]]
    c = bm[:n, :2].astype(np.float64)
    v = np.stack([st[1:n + 1, ix["fwd_v"]], st[1:n + 1, ix["side_v"]]], 1)
    ok = (np.linalg.norm(c, axis=1) >= 0.3) & (np.linalg.norm(v, axis=1) >= 1e-3)
    if ok.sum() < 3:
        return dict(n=int(ok.sum()))
    res = {}
    for name, th in (("c", yaw - yaw0_main), ("b", yaw - st[0, ix["yaw_q"]])):
        x, y = (c[:, 1], c[:, 0]) if ix["fwd_rev"] else (c[:, 0], c[:, 1])
        if ix["fwd_rev"]:
            p = np.stack([x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)], 1)
        else:
            p = np.stack([x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th)], 1)
        ang = np.degrees(np.arctan2(p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0], (p * v).sum(1)))[ok]
        res[name] = round(float(np.median(np.abs(ang))), 2)
        if name == "c":  # signed rotation from the correctly transformed command to the realized velocity
            res["off_c"] = round(float(np.median(ang)), 2)
    return dict(n=int(ok.sum()), **res)


def work(item):
    d, fname, line, row, task, lay, role = item
    try:
        return _work(d, fname, line, row, task, lay, role)
    except Exception as e:
        return dict(dir=d, file=fname, line=line, tag=row.get("tag"), t=row.get("t"), crash=repr(e),
                    tb=traceback.format_exc()[-1500:])


def _work(d, fname, line, row, task, lay, role):
    tag, t = row["tag"], row["t"]
    ix = idx(lay)
    ms = ep_states(task, tag)
    rec = dict(dir=d, file=fname, line=line, task=task, role=role, tag=tag, t=t, k=row.get("k"), L=row.get("L"),
               N=row.get("N"), M=row.get("M"), hidden=row.get("hidden"), k_replay=row.get("k_replay"),
               has_py_state="py_state" in row, py_state_len=len(row["py_state"]) if isinstance(row.get("py_state"), dict) else None,
               episode_success=row.get("episode_success"), instr=row.get("instr"), wall_s=row.get("wall_s"),
               n_cands=len(row["cands"]), n_main=len(ms), main_len=ms.shape[1], state_len=ix["state_len"])
    # main-run quantities at t
    g = lambda s: abs(s[ix["f1"]] - s[ix["f2"]])
    rec["main"] = dict(gap_t=float(g(ms[t])), gap_tm1=float(g(ms[t - 1])) if t else None,
                       dyaw_deg=float(np.degrees(ms[t, ix["yaw_q"]] - ms[0, ix["yaw_q"]])),
                       yaw0=float(ms[0, ix["yaw_q"]]),
                       armv16=float(np.abs(ms[t + 1:t + 17][:, ix["arm_v"]]).max()) if t + 1 < len(ms) else None,
                       objz={o: [float(ms[min(t + s, len(ms) - 1), a + 2]) for s in (0, 4, 8, 16)] for o, a in ix["objs"].items()},
                       time_dev=float(np.abs(np.diff(ms[:, 0]) - 0.05).max()))
    gt, gtm1 = rec["main"]["gap_t"], rec["main"]["gap_tm1"]
    widening = (gt - gtm1 > 0.0005) if gtm1 is not None else False
    rec["main"]["old_rule"] = "closing" if (gt < OLD_OPEN - 0.004 and not widening) else "opening"
    rec["main"]["new_rule"] = "closing" if (gt < ix["open_gap"] - 0.0015 and not widening) else "opening"
    # snap
    sp = f"{DATA}/{d}/snap/{tag}_t{t}.npz"
    snap = dict(exists=os.path.exists(sp))
    if snap["exists"]:
        snap["mtime"] = os.path.getmtime(sp)
        try:
            with zipfile.ZipFile(sp) as zf:
                bad = zf.testzip()
                snap["crc_ok"] = bad is None
                snap["members"] = len(zf.namelist())
                s = np.lib.format.read_array(io.BytesIO(zf.read("state.npy")))
            snap["state_eq_main"] = bool(s.shape == ms[t].shape and np.array_equal(s, ms[t]))
        except Exception as e:
            snap["err"] = repr(e)
    rec["snap"] = snap
    # rollouts
    rolls, arrs = [], {}
    for c in row["cands"]:
        i = c["cand"]
        for r in c["rollouts"]:
            m = r["m"]
            base = f"{DATA}/{d}/roll/{tag}_t{t}_c{i}_m{m}"
            o = dict(i=i, m=m, steps=r["steps"], done=r["done"], success=r.get("success"), reached=r.get("reached"),
                     prev_ok=r.get("prev_ok", True), progress=r.get("progress", 0.0), obs_fp=r.get("obs_fp"),
                     mp4=mp4_ok(base + ".mp4"), schema_missing=[f for f in ("success", "reached", "prev_ok", "progress", "obs_fp") if f not in r])
            if not os.path.exists(base + ".npz"):
                o["npz"] = "missing"; rolls.append(o); continue
            o["mtime"] = os.path.getmtime(base + ".npz")
            try:
                z = load_full(base + ".npz")
            except Exception as e:
                o["npz"] = f"load_error: {e!r}"; rolls.append(o); continue
            o["npz"] = "ok"
            o["keys_missing"] = [k for k in ["states"] + KEYS if k not in z]
            o["keys_extra"] = [k for k in z if k not in ["states"] + KEYS]
            st = z.get("states")
            if st is None:
                rolls.append(o); continue
            o["S"], o["D"] = int(st.shape[0]), int(st.shape[1]) if st.ndim == 2 else None
            o["st_dtype"] = str(st.dtype)
            o["finite"] = bool(np.isfinite(st).all())
            shapes = {k: list(z[k].shape) for k in KEYS if k in z}
            o["C"] = shapes[KEYS[0]][0] if KEYS[0] in shapes else None
            o["shape_bad"] = [k for k, s in shapes.items() if len(s) != 3 or s[1] != 16 or s[2] != DIMS[k] or s[0] != o["C"]]
            o["act_dtype"] = sorted({str(z[k].dtype) for k in KEYS if k in z})
            o["act_finite"] = bool(all(np.isfinite(z[k]).all() for k in KEYS if k in z))
            tt = np.diff(st[:, 0])
            o["time_dev"] = float(np.abs(tt - 0.05).max()) if len(tt) else None
            o["s0_eq_main"] = bool(st.shape[1] == ms.shape[1] and np.array_equal(st[0], ms[t]))
            if not o["s0_eq_main"] and st.shape[1] == ms.shape[1]:
                dd = np.abs(st[0] - ms[t]); o["s0_maxdiff"] = float(dd.max()); o["s0_argmax"] = int(dd.argmax())
            o["h_c0"] = h(np.concatenate([z[k][0].ravel() for k in KEYS])) if o["C"] else None
            o["h_c1"] = h(np.concatenate([z[k][1].ravel() for k in KEYS])) if o["C"] and o["C"] > 1 else None
            o["h_s16"] = h(st[:17])
            o["cm0"] = float(z["action.control_mode"][0, 0, 0])
            o["gcmd0"] = [round(float(x), 3) for x in z["action.gripper_close"][0, :, 0]]
            o["gap"] = [round(float(g(s)), 6) for s in st[:17]]
            av = np.abs(st[1:17][:, ix["arm_v"]]).max(1)
            o["armv16"] = round(float(av.max()), 4) if len(av) else None
            o["armv16_at"] = int(av.argmax()) + 1 if len(av) else None
            bm = z["action.base_motion"].reshape(-1, 4)
            o["base"] = base_dir_test(st, bm, ix, rec["main"]["yaw0"])
            o["base_cmd_max"] = round(float(np.abs(bm[:, :3]).max()), 3)
            o["objz"] = {ob: [round(float(st[min(s, len(st) - 1), a + 2]), 4) for s in (0, 4, 8, 16)] + [round(float(st[-1, a + 2]), 4)]
                         for ob, a in ix["objs"].items()}
            arrs[(i, m)] = (z, st)
            rolls.append(o)
    rec["rolls"] = rolls
    # within-candidate comparisons (continuations share the first chunk)
    pairs = []
    for i in sorted({k[0] for k in arrs}):
        ms_ = sorted(m for (ii, m) in arrs if ii == i)
        for a_ in range(len(ms_)):
            for b_ in range(a_ + 1, len(ms_)):
                (za, sa), (zb, sb) = arrs[(i, ms_[a_])], arrs[(i, ms_[b_])]
                p = dict(i=i, ma=ms_[a_], mb=ms_[b_])
                p["c0_eq"] = all(np.array_equal(za[k][0], zb[k][0]) for k in KEYS)
                p["c0_neq_keys"] = [k.split(".")[1] for k in KEYS if not np.array_equal(za[k][0], zb[k][0])]
                n = min(len(sa), len(sb), 17)
                p["s16_eq"] = bool(np.array_equal(sa[:n], sb[:n]))
                n2 = min(len(sa), len(sb))
                if sa.shape[1] == sb.shape[1]:
                    neq = np.where((sa[:n2] != sb[:n2]).any(1))[0]
                    p["first_div"] = int(neq[0]) if len(neq) else None
                    if p["first_div"] is not None and p["first_div"] <= 16:
                        p["div_max"] = float(np.abs(sa[p["first_div"]] - sb[p["first_div"]]).max())
                p["both_long"] = bool(za[KEYS[0]].shape[0] > 1 and zb[KEYS[0]].shape[0] > 1)
                if p["both_long"]:
                    p["c1_eq"] = all(np.array_equal(za[k][1], zb[k][1]) for k in KEYS)
                    p["c1_eef_eq"] = bool(np.array_equal(za[KEYS[0]][1], zb[KEYS[0]][1]))
                pairs.append(p)
    rec["pairs"] = pairs
    # across candidates: chunk 0 of continuation 0
    c0 = {i: z for (i, m), (z, _) in arrs.items() if m == 0}
    coll = []
    ii = sorted(c0)
    for a_ in range(len(ii)):
        for b_ in range(a_ + 1, len(ii)):
            za, zb = c0[ii[a_]], c0[ii[b_]]
            alleq = all(np.array_equal(za[k][0], zb[k][0]) for k in KEYS)
            eefeq = bool(np.array_equal(za[KEYS[0]][0], zb[KEYS[0]][0]))
            if alleq or eefeq:
                coll.append(dict(a=ii[a_], b=ii[b_], all=alleq, eef=eefeq))
    rec["cand_collisions"] = coll
    rec["n_cand_pairs"] = len(ii) * (len(ii) - 1) // 2
    return rec


def items(only=None, bug1_sample=30):
    sc = scenes()
    lay = json.load(open(os.path.join(HERE, "layout.json")))
    cd = coarse_dirs()
    out = []
    dirs = list(cd) + list(OTHER)
    for d in dirs:
        if only and d not in only:
            continue
        if d in cd:
            task, role = cd[d][0], f"coarse{cd[d][1]}"
        else:
            role, task = OTHER[d][0], OTHER[d][1]
        rows = [(f, i, r) for f, i, r in read_rows(d) if r is not None and "error" not in r and "cands" in r]
        if role == "bug1":  # positive control only: a deterministic sample
            rows = sorted(rows, key=lambda x: hashlib.sha1(f"{x[2]['tag']}{x[2]['t']}".encode()).hexdigest())[:bug1_sample]
        for f, i, r in rows:
            out.append((d, f, i, r, task, lay[sc[r["tag"]]], role))
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 3
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    its = items(only)
    out = os.path.join(HERE, "scan.jsonl" if not only else "scan_partial.jsonl")
    print(f"{len(its)} points, {sum(len(c['rollouts']) for x in its for c in x[3]['cands'])} rollouts", flush=True)
    t0 = time.time(); n = 0
    with open(out, "w") as f, get_context("spawn").Pool(nw) as pool:
        for rec in pool.imap_unordered(work, its, chunksize=1):
            f.write(json.dumps(rec) + "\n"); n += 1
            if n % 100 == 0:
                print(f"{n}/{len(its)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {n} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
