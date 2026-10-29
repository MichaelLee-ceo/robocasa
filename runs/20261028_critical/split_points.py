"""What differs between a winning and a losing candidate chunk at the same decision point?

A split point is a collected decision point (../20261026_critic_data) where at least one candidate finished stage k in
both continuations (a winner) and another in neither (a loser). For every rollout of its winners and losers, the saved
per-step states are replayed without rendering (../20261025_seen480/analyze_stages.py:replay_env, stage_trace), giving:
  * how the rollout went in stage k: first hold, lift, at-target and fixture-touch frames, and for an unfinished stage
    the stage_trace verdict (never_reached, touched_not_grasped, dropped, placed_off_target, ...);
  * the pose where each rollout first holds the stage's object (or first touches its fixture): contact_pose;
  * the pose at the end of the candidate chunk (frame 16; both continuations share it) and at frame 0: end effector
    position, gripper yaw, finger width, base position and yaw, and, for the stage's object, the eef position in the
    object's frame, the eef height above it and the gripper yaw relative to the object's yaw.
The object of a pick-and-place stage is the one a winner holds first in the stage (else the one nearest the eef).
Labels, rollouts and py_state all come from the coarse row in use (data.py), so a re-collected point is replayed from
its own rollouts. Writes split_points.jsonl (one row per split point) to $CRIT_OUT. usage: python -I split_points.py <workers>"""
import json, multiprocessing, os, sys
import numpy as np

R = os.environ["ROBOCASA_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", HERE]
import data  # noqa: E402
from data import OUT  # noqa: E402


def yaw(m):
    m = np.asarray(m).reshape(3, 3)
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def set_py_state(env, ps):
    """python-side fixture state the sim state does not hold (../20261026_imagine/imagine.py:set_py_state)."""
    for n, d in ps.items():
        for k, v in d.items():
            setattr(env.fixtures[n], k, v)


def pose(env, task, k, obj):
    rob = env.robots[0]
    site = rob.eef_site_id["right"]
    eef, R_eef = np.array(env.sim.data.site_xpos[site]), np.array(env.sim.data.site_xmat[site]).reshape(3, 3)
    base_id = env.sim.model.body_name2id(rob.robot_model.root_body)
    out = dict(eef=eef.round(4).tolist(), eef_yaw=round(yaw(R_eef), 1), eef_down=round(float(-R_eef[2, 2]), 3),
               base=np.array(env.sim.data.body_xpos[base_id])[:2].round(4).tolist(), base_yaw=round(yaw(env.sim.data.body_xmat[base_id]), 1),
               width=round(float(sum(abs(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(j)]) for j in rob.gripper["right"].joints)), 4))
    if obj is not None:
        b = env.obj_body_id[obj]
        p, Ro = np.array(env.sim.data.body_xpos[b]), np.array(env.sim.data.body_xmat[b]).reshape(3, 3)
        out.update(obj=obj, eef_in_obj=(Ro.T @ (eef - p)).round(4).tolist(), dist=round(float(np.linalg.norm(eef - p)), 4),
                   height=round(float(eef[2] - p[2]), 4), yaw_rel=round(float((yaw(R_eef) - yaw(Ro) + 90) % 180 - 90), 1))
    return out


def rollout_summary(rec, task, k):
    from stage_trace import STAGE_SPECS, analyze, _holds
    sp = STAGE_SPECS[task][k]
    T = len(rec["stage"])
    s = analyze(rec, task)["stages"][k]
    out = dict(T=T, first_true=s["first_true"], verdict=[d.get("verdict") for d in s.get("details", [])] if isinstance(s.get("details"), list)
               else ([s["details"].get("verdict")] if s.get("details") else []))
    if sp["kind"] == "pnp":
        holds = {o: _holds(rec["obj"][o], 0, T) for o in sp["objs"]}
        first = min(((iv[0][0], o) for o, iv in holds.items() if iv), default=(None, None))
        out.update(first_hold=first[0], hold_obj=first[1], n_holds=sum(len(iv) for iv in holds.values()))
        if first[1] is not None:
            r = rec["obj"][first[1]]
            z0 = r["pos"][0][2]
            out["first_lift"] = next((f for f in range(T) if r["pos"][f][2] > z0 + 0.02 and r["touch"][f]), None)
    elif sp["kind"] == "fixture":
        t = rec["fx_touch"][sp["fixture"]]
        out.update(first_fx_touch=next((f for f in range(T) if t[f]), None), fx_value_max=max(rec["fx_value"][k]) if rec["fx_value"][k] else None)
    return out


def work(job):
    import atomic_eval  # noqa: F401  (registers the robocasa gym envs, as in the rollout)
    from oracle import reset_to, set_state
    from stage_trace import STAGE_SPECS, trace_episode
    from analyze_stages import replay_env
    p, roll = job
    task, k = p["task"], p["k"]
    try:
        z = np.load(p["scene"], allow_pickle=True)
        env = replay_env(task)
        reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
        sp = STAGE_SPECS[task][k]
        cands, recs = {}, {}
        for i in p["winners"] + p["losers"]:
            for m in (0, 1):
                states = np.load(f"{roll}/{p['tag']}_t{p['t']}_c{i}_m{m}.npz")["states"]
                set_py_state(env, p["py_state"])
                recs[(i, m)] = (trace_episode(env, task, states, set_state), states)
        obj = None
        if sp["kind"] == "pnp":
            firsts = [rollout_summary(recs[(i, m)][0], task, k) for i in p["winners"] for m in (0, 1)]
            held = sorted((f["first_hold"], f["hold_obj"]) for f in firsts if f.get("first_hold") is not None)
            if held:
                obj = held[0][1]
            else:
                rec0 = recs[(p["winners"][0], 0)][0]
                obj = min(sp["objs"], key=lambda o: rec0["obj"][o]["eef_dist"][0])
        for i in p["winners"] + p["losers"]:
            rows = [rollout_summary(recs[(i, m)][0], task, k) for m in (0, 1)]
            for m, row in zip((0, 1), rows):  # the pose where the rollout first holds the object / touches the fixture
                f = row.get("first_hold", row.get("first_fx_touch"))
                if f is not None:
                    set_state(env, recs[(i, m)][1][f]); row["contact_pose"] = pose(env, task, k, row.get("hold_obj") or obj)
            states = recs[(i, 0)][1]
            poses = {}
            for f in (0, 16, 48):
                if f < len(states):
                    set_state(env, states[f]); poses[f] = pose(env, task, k, obj)
            cands[i] = dict(win=i in p["winners"], rollouts=rows, pose=poses)
        return dict(task=task, tag=p["tag"], t=p["t"], k=k, kind=sp["kind"], obj=obj, winners=p["winners"], losers=p["losers"], cands=cands)
    except Exception as e:
        import traceback
        return dict(task=task, tag=p["tag"], t=p["t"], k=k, error=repr(e), tb=traceback.format_exc()[-1500:])


def split_points():
    """(job, roll dir) per split point; the labels, roll dir and py_state are those of the coarse row in use"""
    from effect_by_event import points
    jobs = []
    for p in points():
        per = p["y"].sum(1)
        win, lose = [int(i) for i in np.nonzero(per == 2)[0]], [int(i) for i in np.nonzero(per == 0)[0]]
        if win and lose:
            jobs.append((dict(task=p["task"], tag=p["tag"], t=p["t"], k=p["k"], winners=win, losers=lose,
                              scene=data.scenes()[p["tag"]], py_state=p["py_state"]), p["roll"]))
    return jobs


if __name__ == "__main__":
    jobs = split_points()
    if os.environ.get("N"):  # smoke test: the first N split points
        jobs = jobs[:int(os.environ["N"])]
    print(f"{len(jobs)} split points", flush=True)
    with multiprocessing.get_context("spawn").Pool(int(sys.argv[1])) as pool, open(os.path.join(OUT, "split_points.jsonl"), "w") as f:
        for n, res in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            f.write(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n"); f.flush()
            if "error" in res or n % 20 == 0:
                print(n, res["tag"], res["t"], res.get("error", ""), flush=True)
