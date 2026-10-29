"""Bug 4 check: settle() infers the gripper's commanded side from the finger gap. A wide grasp of a big object (gap
within 4 mm of fully open) was read as "opening", so the restored gripper ramp started at "open" and the object could
fall (SteamInMicrowave s4 t160: the vegetable back in the sink within 8 steps in all 16 rollouts).

For every collected row of every dataset and layer (data.all_rows: coarse, re-collected, dense, dense2, confirm, the
superseded rows included, each flagged in_use) where the main run's fingers touch an object at t (its trace), this
reports what the old rule inferred and whether the held object fell in the first 8 steps of the row's own rollout c0 m0
(>= 10 cm below the main run's). Writes grip_check.json to $CRIT_OUT. usage: python -I grip_check.py <workers>"""
import gzip, json, multiprocessing, os, sys
import numpy as np

R = os.environ["ROBOCASA_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", HERE]
import data  # noqa: E402
SEEN = f"{R}/runs/20261025_seen480"


def work(job):
    import atomic_eval  # noqa: F401
    from oracle import reset_to, set_state
    from analyze_stages import replay_env
    task, tag, scene, pts = job
    try:
        env = replay_env(task)
        z = np.load(scene, allow_pickle=True); reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
        rob = env.robots[0]; gi = rob._ref_gripper_joint_pos_indexes["right"]; g = rob.gripper["right"]
        open_gap = abs(g.init_qpos[0] - g.init_qpos[1]) * 1.9
        ms = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
        gap = lambda s: abs(s[1 + gi[0]] - s[1 + gi[1]])
        out = []
        for t, obj, d, hidden, ds, use in pts:
            closing = gap(ms[t]) < open_gap - 0.004 and not (gap(ms[t]) - gap(ms[t - 1]) > 0.0005 if t else False)
            b = env.obj_body_id[obj]
            set_state(env, ms[min(t + 8, len(ms) - 1)]); zm = float(env.sim.data.body_xpos[b][2])
            st = np.load(f"{d}/{tag}_t{t}_c0_m0.npz")["states"]
            set_state(env, st[min(8, len(st) - 1)]); zr = float(env.sim.data.body_xpos[b][2])
            out.append(dict(task=task, tag=tag, t=t, obj=obj, dir=data.rel(os.path.dirname(d)), dataset=ds, in_use=use, hidden=hidden,
                            gap_mm=round(1000 * gap(ms[t]), 1), open_gap_mm=round(1000 * open_gap, 1), inferred="closing" if closing else "opening",
                            fell=bool(zm - zr >= 0.10)))
        return out
    except Exception as e:
        import traceback
        return [dict(tag=tag, error=repr(e), tb=traceback.format_exc()[-800:])]


def jobs():
    by = {}
    for task in data.TASKS:
        for ds in data.DATASETS:
            for r in data.all_rows(task, ds):
                by.setdefault((task, r["tag"]), []).append((r["t"], data.roll(r), r.get("hidden"), ds, r["_in_use"]))
    out = []
    for (task, tag), pts in by.items():
        tr = json.load(gzip.open(f"{SEEN}/analysis/{task}/{tag}_trace.json.gz"))
        keep = []
        for t, d, hidden, ds, use in pts:
            held = [o for o, r in tr["obj"].items() if t < len(r["touch"]) and r["touch"][t]]  # touching: a wide grasp of a big object is often not "grasped" for OU.check_obj_grasped
            if held:
                keep.append((t, held[0], d, hidden, ds, use))
        if keep:
            out.append((task, tag, data.scenes()[tag], keep))
    return out


if __name__ == "__main__":
    js = jobs()
    print(f"{sum(len(j[3]) for j in js)} held-object points in {len(js)} episodes", flush=True)
    rows = []
    with multiprocessing.get_context("spawn").Pool(int(sys.argv[1])) as pool:
        for r in pool.imap_unordered(work, js):
            rows += r
    json.dump(rows, open(os.path.join(data.OUT, "grip_check.json"), "w"), indent=0)
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"{len(bad)} errors", bad[:1])
    for task in data.TASKS:
        for lab, g in [("all rows", [r for r in rows if r["task"] == task]), ("rows in use", [r for r in rows if r["task"] == task and r["in_use"]])]:
            op = [r for r in g if r["inferred"] == "opening"]
            print(f"{task:21s} {lab:11s} held points {len(g):4d}: inferred opening {len(op):4d} (fell at once {sum(r['fell'] for r in op)}); "
                  f"inferred closing {len(g) - len(op)} (fell at once {sum(r['fell'] for r in g if r['inferred'] == 'closing')})")
        g = [r for r in rows if r["task"] == task]
        for d in sorted({r["dir"] for r in g}):
            gg = [r for r in g if r["dir"] == d]
            print(f"    {d:32s} held {len(gg):4d} (in use {sum(r['in_use'] for r in gg):4d}), inferred opening "
                  f"{sum(r['inferred'] == 'opening' for r in gg):4d}, fell {sum(r['fell'] for r in gg)}")
