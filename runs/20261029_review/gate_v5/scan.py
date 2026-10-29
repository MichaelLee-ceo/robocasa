"""Physics-anomaly scan, step 1: extract per-step physical signals of the main runs and of the rollouts' first 32 steps.

For every episode (tag) that has collected decision points, one worker loads the scene once (no rendering;
../../20261025_seen480/analyze_stages.py:replay_env + oracle.reset_to), then for every saved state it sets the flattened
sim state and runs mj_kinematics only (positions; ~0.1 ms) and, every 4th step, mj_fwdPosition (contacts). It writes
feat/<tag>.pkl with:
  main: the whole main run, every state: eef (world and base frame), base x/y/yaw, finger gap, object positions,
        robot / object / fixture joint velocities; gripper-object contacts every 4th state; the trace's touch.
  pts:  one record per collected point instance (dir, role, hidden, t, k, labels) with its rollouts' first 33 states
        (start + 32 steps) reduced the same way, plus the first two chunks' actions and the start-state check.
Roles: cur (current coarse row, effect_by_event.points() order), sup (a superseded coarse row), dense, dense_bug3,
dense2 (complete rows only), confirm, bug1 (run/, run_SteamInMicrowave/: before the 2026-10-27 fix).
usage: python -I scan.py <workers> [m_max] [tag regex]"""
import glob, gzip, json, multiprocessing, os, pickle, re, sys, time, traceback
import numpy as np

R = os.environ["ROBOCASA_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480"]
DATA = f"{R}/runs/20261026_critic_data"
SEEN = f"{R}/runs/20261025_seen480"
OUT = os.path.join(HERE, "feat")  # this folder's own features (gate_v5)
NS = 33  # start + 32 steps
CSTEP = 4  # contacts every 4th state

# coarse dirs in effect_by_event.points() order: later rows replace earlier ones for the same (tag, t)
# gate_v5 (2026-10-29): the settled-v5 dirs added; a fix5 row replaces the coarse row of its (tag, t) ("cur"; the
# replaced row becomes "sup"), and the dense/dense2 fix5 rows get roles dense_v5 / dense2_v5 next to the rows they replace.
COARSE = {"KettleBoiling": ["run_v2/s*", "run_KettleBoiling_fix", "run_KettleBoiling_fix3", "run_KettleBoiling_fix5"],
          "SteamInMicrowave": ["run_SteamInMicrowave_v2", "run_SteamInMicrowave_fix", "run_SteamInMicrowave_fix3", "run_SteamInMicrowave_fix5"],
          "PackIdenticalLunches": ["run_PackIdenticalLunches", "run_PackIdenticalLunches_fix", "run_PackIdenticalLunches_fix3", "run_PackIdenticalLunches_fix5"]}
OTHER = [("run_dense_KettleBoiling", "dense"), ("run_dense_SteamInMicrowave", "dense"), ("run_dense_PackIdenticalLunches", "dense"),
         ("run_dense_KettleBoiling_fix5", "dense_v5"), ("run_dense_SteamInMicrowave_fix5", "dense_v5"), ("run_dense_PackIdenticalLunches_fix5", "dense_v5"),
         ("run_dense2_SteamInMicrowave", "dense2"), ("run_dense2_PackIdenticalLunches", "dense2"),
         ("run_dense2_SteamInMicrowave_fix5", "dense2_v5"), ("run_dense2_PackIdenticalLunches_fix5", "dense2_v5")]
SCENE_JOBS = ["jobs.json", "jobs_SteamInMicrowave.json", "jobs_PackIdenticalLunches.json"]


def rows(pattern):
    out = []
    for p in sorted(glob.glob(f"{DATA}/{pattern}/results_w*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line a worker is still writing
            if "error" in r:
                continue
            r["_dir"] = os.path.relpath(os.path.dirname(p), DATA)
            out.append(r)
    return out


def slim(r, role):
    return dict(dir=r["_dir"], role=role, hidden=r.get("hidden"), tag=r["tag"], task=r["task"], t=r["t"], k=r["k"], L=r["L"],
                N=r["N"], M=r["M"], labels=[[dict(done=x["done"], steps=x["steps"]) for x in c["rollouts"]] for c in r["cands"]])


def inventory():
    pts = []
    for task, dirs in COARSE.items():
        last = {}
        for d in dirs:
            for r in rows(d):
                key = (r["tag"], r["t"])
                if key in last:
                    pts.append(slim(last[key], "sup"))
                last[key] = r
        pts += [slim(r, "cur") for r in last.values()]
    for d, role in OTHER:
        pts += [slim(r, role) for r in rows(d)]
    return pts


def layout(env):
    """indices into the flattened state and the model for the robot, the objects and the fixtures."""
    m = env.sim.model
    nq, nv = m.nq, m.nv
    jid = lambda n: m.joint_name2id(n)
    rnames = ["mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw",
              "mobilebase0_joint_torso_height"] + [f"robot0_joint{i}" for i in range(1, 8)] + \
             ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
    rob_dof = np.array([m.jnt_dofadr[jid(n)] for n in rnames])
    objs = list(env.obj_body_id.keys())
    obj_bid = np.array([env.obj_body_id[o] for o in objs])
    obj_dof = []
    for b in obj_bid:
        j = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == 0]
        obj_dof.append(m.jnt_dofadr[j[0]] if j else -1)
    obj_dof = np.array(obj_dof)
    used = set(rob_dof.tolist())
    for d in obj_dof:
        if d >= 0:
            used |= set(range(d, d + 6))
    fix_dof = np.array([d for d in range(nv) if d not in used])
    # gripper geoms and object geoms (for contacts)
    gripper_geoms = np.array([g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("gripper0_")])
    root = np.array(m.body_rootid)
    geom_obj = np.full(m.ngeom, -1)
    for i, b in enumerate(obj_bid):
        geom_obj[root[m.geom_bodyid] == root[b]] = i
    is_grip = np.zeros(m.ngeom, bool); is_grip[gripper_geoms] = True
    rob = env.robots[0]
    return dict(nq=nq, nv=nv, rob_dof=rob_dof, objs=objs, obj_bid=obj_bid, obj_dof=obj_dof, fix_dof=fix_dof,
                fix_names=[m.joint_id2name(m.dof_jntid[d]) for d in fix_dof], geom_obj=geom_obj, is_grip=is_grip,
                base_bid=m.body_name2id("mobilebase0_base"), eef_site=rob.eef_site_id["right"],
                gi=np.array(rob._ref_gripper_joint_pos_indexes["right"]))


def reduce(env, L, states, contacts_every=CSTEP):
    """per-state physical signals of a sequence of flattened states."""
    import mujoco
    m, d = env.sim.model._model, env.sim.data._data
    S = len(states)
    nq = L["nq"]
    out = dict(T=S, time=states[:, 0].astype(np.float64),
               eef_w=np.zeros((S, 3), np.float32), eef_b=np.zeros((S, 3), np.float32), base=np.zeros((S, 3), np.float32),
               obj=np.zeros((S, len(L["objs"]), 3), np.float32), obj_rot=np.zeros((S, len(L["objs"]), 3, 3), np.float16))
    q = states[:, 1:1 + nq]
    v = states[:, 1 + nq:1 + nq + L["nv"]]
    out["gap"] = np.abs(q[:, L["gi"][0]] - q[:, L["gi"][1]]).astype(np.float32)
    out["qv_rob"] = v[:, L["rob_dof"]].astype(np.float32)
    od = L["obj_dof"]
    out["qv_obj"] = np.stack([v[:, a:a + 6] if a >= 0 else np.zeros((S, 6)) for a in od], 1).astype(np.float32)
    fv = np.abs(v[:, L["fix_dof"]]) if len(L["fix_dof"]) else np.zeros((S, 1))
    out["qv_fix_max"] = fv.max(1).astype(np.float32)
    out["qv_fix_arg"] = fv.argmax(1).astype(np.int16)
    tsteps = list(range(0, S, contacts_every))
    touch = np.zeros((len(tsteps), len(L["objs"])), bool)
    ci = 0
    for s in range(S):
        d.time = states[s, 0]; d.qpos[:] = q[s]; d.qvel[:] = v[s]
        mujoco.mj_kinematics(m, d)
        if contacts_every and s % contacts_every == 0:
            mujoco.mj_fwdPosition(m, d)
            n = d.ncon
            if n:
                g1 = d.contact.geom1[:n]; g2 = d.contact.geom2[:n]
                gg = L["is_grip"]
                o1 = L["geom_obj"][g2[gg[g1]]]; o2 = L["geom_obj"][g1[gg[g2]]]
                for o in np.concatenate([o1, o2]):
                    if o >= 0:
                        touch[ci, o] = True
            ci += 1
        bp = d.xpos[L["base_bid"]]; bR = d.xmat[L["base_bid"]].reshape(3, 3)
        e = d.site_xpos[L["eef_site"]]
        out["eef_w"][s] = e; out["eef_b"][s] = bR.T @ (e - bp)
        out["base"][s] = (bp[0], bp[1], np.arctan2(bR[1, 0], bR[0, 0]))
        out["obj"][s] = d.xpos[L["obj_bid"]]
        out["obj_rot"][s] = d.xmat[L["obj_bid"]].reshape(-1, 3, 3)
    out["touch"] = touch
    out["touch_steps"] = np.array(tsteps)
    return out


def build(task, scene):
    import analyze_stages
    from oracle import reset_to
    last = None
    for attempt in range(2):
        try:
            env = analyze_stages.replay_env(task)
            z = np.load(scene, allow_pickle=True)
            reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
            return env
        except Exception as e:  # "could not initialize task": retry once with a fresh env
            last = e
            for v in analyze_stages._ENV.values():
                try:
                    v.close()
                except Exception:
                    pass
            analyze_stages._ENV.clear()
    raise last


def work(job):
    import atomic_eval  # noqa: F401  (registers the robocasa gym envs, as in the rollout)
    task, tag, scene, pts, m_max = job
    t0 = time.time()
    try:
        env = build(task, scene)
        L = layout(env)
        ms = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
        if ms.shape[1] != 1 + L["nq"] + L["nv"]:
            raise RuntimeError(f"state length {ms.shape[1]} vs model {1 + L['nq'] + L['nv']}")
        tr = json.load(gzip.open(f"{SEEN}/analysis/{task}/{tag}_trace.json.gz"))
        main = reduce(env, L, ms)
        main["trace_touch"] = np.array([[bool(tr["obj"][o]["touch"][s]) if o in tr["obj"] and s < len(tr["obj"][o]["touch"]) else False
                                         for o in L["objs"]] for s in range(len(ms))])
        main["trace_grasp"] = np.array([[bool(tr["obj"][o]["grasp"][s]) if o in tr["obj"] and s < len(tr["obj"][o]["grasp"]) else False
                                         for o in L["objs"]] for s in range(len(ms))])
        recs, nroll, missing = [], 0, []
        for p in pts:
            rr = []
            for c in range(p["N"]):
                for mm in range(min(p["M"], m_max)):
                    f = f"{DATA}/{p['dir']}/roll/{tag}_t{p['t']}_c{c}_m{mm}.npz"
                    if not os.path.exists(f):
                        missing.append(f); continue
                    z = np.load(f)
                    st = z["states"][:NS]
                    r = reduce(env, L, st)
                    r.update(c=c, m=mm, start_diff=float(np.abs(st[0] - ms[p["t"]]).max()) if p["t"] < len(ms) else None,
                             n_states=int(len(z["states"])), **p["labels"][c][mm])
                    for k in ("control_mode", "gripper_close", "base_motion", "end_effector_position", "end_effector_rotation"):
                        r["a_" + k] = z["action." + k][:2].astype(np.float32)
                    rr.append(r); nroll += 1
            recs.append(dict(p, rolls=rr))
        out = dict(task=task, tag=tag, objs=L["objs"], fix_names=L["fix_names"], main=main, pts=recs, missing=missing)
        with open(f"{OUT}/{tag}.pkl.tmp", "wb") as fh:
            pickle.dump(out, fh, protocol=5)
        os.replace(f"{OUT}/{tag}.pkl.tmp", f"{OUT}/{tag}.pkl")
        return dict(tag=tag, n_pts=len(pts), n_roll=nroll, missing=len(missing), s=round(time.time() - t0, 1))
    except Exception as e:
        return dict(tag=tag, error=repr(e), tb=traceback.format_exc()[-1500:], s=round(time.time() - t0, 1))


def jobs(m_max, rx):
    scenes = {}
    for f in SCENE_JOBS:
        for j in json.load(open(f"{DATA}/{f}")):
            scenes[j["tag"]] = j["scene"]
    by = {}
    for p in inventory():
        by.setdefault((p["task"], p["tag"]), []).append(p)
    order = {"KettleBoiling": 0, "SteamInMicrowave": 1, "PackIdenticalLunches": 2}
    out = [(task, tag, scenes[tag], pts, m_max) for (task, tag), pts in by.items() if re.search(rx, tag)]
    out.sort(key=lambda j: (order[j[0]], j[1]))
    return out


if __name__ == "__main__":
    nw = int(sys.argv[1]); m_max = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    rx = sys.argv[3] if len(sys.argv) > 3 else "."
    os.makedirs(OUT, exist_ok=True)
    js = jobs(m_max, rx)
    js = [j for j in js if not os.path.exists(f"{OUT}/{j[1]}.pkl")]  # resumable
    print(f"{len(js)} episodes, {sum(len(j[3]) for j in js)} point instances, m < {m_max}", flush=True)
    t0 = time.time()
    with multiprocessing.get_context("spawn").Pool(nw, maxtasksperchild=None) as pool:
        for r in pool.imap_unordered(work, js):
            print(json.dumps(r), f"elapsed {time.time() - t0:.0f}s", flush=True)
