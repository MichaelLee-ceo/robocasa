"""Physics-anomaly scan, step 3: flags, rates and the exchangeability (rank) test over feat/*.pkl.

Writes (all in this folder):
  rollouts.csv.gz  one row per scanned rollout: point, role, candidate, labels, flags vs the main run (M_*) and vs a
                   sibling candidate (S_*), details (base angle, eef speed, joint velocities, finger events, ...)
  mainwin.csv.gz   one row per main-run window: at every collected point (kind=point) and at every chunk boundary of
                   the 90 main runs (kind=chunk), with the absolute flags of the main run itself and its stage
  ranks.csv.gz     per point instance and feature group: the main run's outlier rank among itself + the 8 m0
                   candidates (1 = most typical, 9 = most outlying; uniform under a faithful restore)
usage: python -I analyze.py [workers]"""
import glob, json, multiprocessing, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import flags, traj_stats, window, finger_events, unwrap_dyaw  # noqa: E402

R = os.environ.get("ROBOCASA_ROOT", "/home/michaellee/mclee/robocasa")
SEEN = f"{R}/runs/20261025_seen480"
ABS = ("F1b", "F1c", "F3a", "F3b", "F5")
REL = ("F1a", "F1r", "F2", "F4", "F4e")


def histories():
    h = {}
    for p in glob.glob(f"{SEEN}/run/results_w*.jsonl"):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tag" in r:
                h[r["tag"]] = r.get("history", [])
            elif "job" in r and isinstance(r["job"], dict):
                h[r["job"].get("tag")] = r.get("history", [])
    return h


def stage_at(hist, s):
    return sum(1 for tf, _ in hist if tf <= s)


def held_obj(main, t):
    tt = main["trace_touch"][t]
    return int(np.argmax(tt)) if tt.any() else None


def feat_groups(tr, n, oi):
    """feature vectors of a trajectory for the rank test (displacements from its own start state)."""
    S = [s for s in (1, 2, 4, 8, 16) if s < n]
    g = {}
    g["eef"] = np.concatenate([tr["eef_b"][s] - tr["eef_b"][0] for s in S])
    g["base"] = np.concatenate([np.r_[tr["base"][s, :2] - tr["base"][0, :2], unwrap_dyaw(tr["base"][0, 2], tr["base"][s, 2])] for s in S])
    g["grip"] = np.array([tr["gap"][s] - tr["gap"][0] for s in S])
    g["early"] = np.concatenate([np.r_[tr["eef_b"][s] - tr["eef_b"][0], tr["base"][s, :2] - tr["base"][0, :2], tr["gap"][s] - tr["gap"][0]]
                                 for s in (1, 2) if s < n])
    if oi is not None:
        g["obj"] = np.concatenate([tr["obj"][s, oi] - tr["obj"][0, oi] for s in S if s >= 4])
    g["arm_qv"] = np.array([np.abs(tr["qv_rob"][:n, 4:11]).max()])
    return g


def outlier_rank(X):
    """X: (9, d), row 0 the main run. Leave-one-out distance of each row to the mean of the others, every dim scaled by
    the spread of the 9 (MAD, floored); returns the main run's rank (1 = most typical .. 9 = most outlying), its
    distance and the median distance of the candidates."""
    med = np.median(X, 0)
    sc = np.maximum(1.4826 * np.median(np.abs(X - med), 0), np.maximum(1e-4, 0.1 * np.abs(X).max(0)))
    Z = X / sc
    d = np.array([np.linalg.norm(Z[j] - np.delete(Z, j, 0).mean(0)) for j in range(len(Z))])
    return int((d < d[0]).sum() + 1 + 0.5 * (d[1:] == d[0]).sum()), float(d[0]), float(np.median(d[1:]))


def process(path):
    import pickle
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    task, tag, main, objs = d["task"], d["tag"], d["main"], d["objs"]
    hist = HIST.get(tag, [])
    T = main["T"]
    base0 = main["base"][0, 2]
    recs, mw, rk = [], [], []
    # main-run windows at every chunk boundary (its own behaviour at every phase)
    for t in range(0, T - 17, 16):
        w = window(main, t)
        n = w["T"]
        st = traj_stats(w, n)
        ev, _ = finger_events(w, n, use_trace=True)
        mw.append(dict(kind="chunk", task=task, tag=tag, t=t, k=stage_at(hist, t), n=n, F1b=bool((st["drop"] > 0.10).any()),
                       F1c=bool((st["v_obj"] > 2.0).any()), F3a=st["eef_v"] > 1.0, F3b=st["eef_jump"], eef_v=st["eef_v"],
                       arm_qv=st["arm_qv"], fing_qv=st["fing_qv"], fix_qv=st["fix_qv"], obj_w=st["obj_w"], gap8=st["gap8"],
                       F4ev=bool(ev), F4ev0=any(s == 0 for s, _ in ev), touch0=bool(w["trace_touch"][0].any()),
                       base_d=float(np.linalg.norm(st["base_d"])), dyaw=st["dyaw"]))
    for p in d["pts"]:
        t = p["t"]
        if t >= T - 1:
            continue
        w = window(main, t)
        oi = held_obj(main, t)
        dyaw_ep = unwrap_dyaw(base0, main["base"][t, 2])
        m0 = {r["c"]: r for r in p["rolls"] if r["m"] == 0}
        info = dict(task=task, tag=tag, t=t, k=p["k"], dir=p["dir"].split("/")[0], role=p["role"], hidden=p["hidden"],
                    dyaw_ep=round(float(np.degrees(dyaw_ep)), 1), held=objs[oi] if oi is not None else None,
                    held_grasp=bool(main["trace_grasp"][t].any()), gap0=float(main["gap"][t]))
        for r in p["rolls"]:
            n = min(r["T"], w["T"])
            if n < 2:
                continue
            fm, det = flags(r, w, n, ref_trace=True)
            sib = m0.get((r["c"] + 1) % p["N"])
            fs = None
            if sib is not None and sib["T"] >= 2:
                fs, dets = flags(r, sib, min(r["T"], sib["T"]))
            cm = r["a_control_mode"][0, :, 0]
            gc = r["a_gripper_close"][0, :, 0]
            rec = dict(info, c=r["c"], m=r["m"], done=r["done"], steps=r["steps"], n=n, start_diff=r["start_diff"],
                       mode0=float(cm[0] > 0.5), mode_frac=float((cm > 0.5).mean()), close0=float(gc[0] > 0.5),
                       close_frac8=float((gc[:8] > 0.5).mean()),
                       **{f"M_{k}": v for k, v in fm.items()}, **({f"S_{k}": v for k, v in fs.items()} if fs else {}))
            for k in ("base_ang", "base_d", "base_d_ref", "dyaw", "dyaw_ref", "eef_v", "arm_qv", "fing_qv", "fix_qv", "obj_w", "obj_v",
                      "gap8", "gap8_ref"):
                rec[k] = det[k]
            rec["fing_ev"] = json.dumps([(int(s), round(o * 1000, 1)) for s, o in det["fing_ev"]])
            rec["fing_ev_ref"] = json.dumps([(int(s), round(o * 1000, 1)) for s, o in det["fing_ev_ref"]])
            rec["obj_move"] = ",".join(objs[i] for i in det["obj_move"])
            rec["obj_fall"] = ",".join(objs[i] for i in det["obj_fall"])
            rec["drop_max"] = float(det["drop"].max()); rec["drop_ref_max"] = float(det["drop_ref"].max())
            if oi is not None:
                rec["held_drop"] = float(det["drop"][oi]); rec["held_drop_ref"] = float(det["drop_ref"][oi])
                rec["held_dz16"] = float(r["obj"][min(16, n - 1), oi, 2] - w["obj"][min(16, n - 1), oi, 2])
            # early divergence from the main run
            for s in (1, 2, 4, 8):
                if s < n:
                    rec[f"de_eef{s}"] = float(np.linalg.norm(r["eef_b"][s] - w["eef_b"][s]))
                    rec[f"de_gap{s}"] = float(r["gap"][s] - w["gap"][s])
                    rec[f"de_base{s}"] = float(np.linalg.norm(r["base"][s, :2] - w["base"][s, :2]))
            recs.append(rec)
        # main window at this point (absolute flags)
        n = w["T"]
        st = traj_stats(w, n)
        ev, _ = finger_events(w, n, use_trace=True)
        mw.append(dict(info, kind="point", n=n, F1b=bool((st["drop"] > 0.10).any()), F1c=bool((st["v_obj"] > 2.0).any()),
                       F3a=st["eef_v"] > 1.0, F3b=st["eef_jump"], eef_v=st["eef_v"], arm_qv=st["arm_qv"], fing_qv=st["fing_qv"],
                       fix_qv=st["fix_qv"], obj_w=st["obj_w"], gap8=st["gap8"], F4ev=bool(ev), F4ev0=any(s == 0 for s, _ in ev),
                       touch0=bool(w["trace_touch"][0].any()), base_d=float(np.linalg.norm(st["base_d"])), dyaw=st["dyaw"]))
        # rank test: main + the m0 rollout of every candidate
        if len(m0) == p["N"] == 8:
            trs = [w] + [m0[c] for c in range(8)]
            for L in (17, 33):
                n = min(min(x["T"] for x in trs), L)
                if n < 3:
                    continue
                G = [feat_groups(x, n, oi) for x in trs]
                for g in G[0]:
                    X = np.stack([x[g] for x in G])
                    rank, dm, dc = outlier_rank(X)
                    rk.append(dict(info, L=L, n=n, group=g, rank=rank, d_main=dm, d_cand=dc))
    return recs, mw, rk


def init():
    global HIST
    HIST = histories()


if __name__ == "__main__":
    import pandas as pd
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    fs = sorted(glob.glob(os.path.join(HERE, "feat", "*.pkl")))
    R1, R2, R3 = [], [], []
    with multiprocessing.get_context("spawn").Pool(nw, initializer=init) as pool:
        for a, b, c in pool.imap_unordered(process, fs):
            R1 += a; R2 += b; R3 += c
    pd.DataFrame(R1).to_csv(os.path.join(HERE, "rollouts.csv.gz"), index=False)
    pd.DataFrame(R2).to_csv(os.path.join(HERE, "mainwin.csv.gz"), index=False)
    pd.DataFrame(R3).to_csv(os.path.join(HERE, "ranks.csv.gz"), index=False)
    print(len(R1), "rollouts,", len(R2), "main windows,", len(R3), "rank rows from", len(fs), "episodes")
