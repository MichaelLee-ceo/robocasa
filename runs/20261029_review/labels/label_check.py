"""Task 2: re-evaluate the labels of a stratified sample of rollouts from their saved states, without rendering.

Per decision point (as collect.py:decision_point): the scene is loaded with atomic_eval.load_scene (scene fingerprint
checked) into the no-render env of analyze_stages.replay_env; S and k_replay are rebuilt with collect.replay on the main
run's saved states 0..t; py_state(env) after the replay is compared with the row's py_state; progress_ref at t.
Per rollout (as atomic_eval.rollout): set_py_state(row py_state), set_state(states[0]) (== the main run's state at t?),
then every saved step i: set_state(states[i]) (fixtures' update_state runs, as in each env step), reward = env._check_success()
(the gym wrapper's per-step success); at every chunk boundary (i % 16 == 0, or the last saved step) the stage loop
`while k < len(steps) and steps[k].check(env, S): k += 1`, then stop if success or k >= k_stop. Then done, prev_ok,
steps, success, reached, progress are recomputed exactly as collect.py does and compared with the stored labels.
Also: the stage-k check at every step (on a copy of S; flags a condition that held between boundaries only), the
conjuncts of the earlier stages when prev_ok is False, and where the rollout should have stopped.

Known limit: online, the checks at a step run on the kinematics/contacts that MuJoCo computed before the last substep's
integration (lite_physics: step1, then step2 integrates qpos without a new forward); set_state + forward evaluates them
on the saved (post-integration) qpos, one 2 ms substep later. A label can therefore flip only when a predicate is within
one substep of its threshold; resim.py re-simulates such rollouts from their actions to settle them.
usage: python label_check.py sample.json out.jsonl <workers>"""
import copy, json, multiprocessing, os, sys, time
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261028_critical", f"{R}/runs/20261026_imagine",
                f"{R}/runs/20261026_critic_data", os.path.dirname(os.path.abspath(__file__))]
SEEN = f"{R}/runs/20261025_seen480"


def conj_sim(e):
    import robocasa.utils.object_utils as OU
    from atomic_steps import _released
    gp = e.sim.data.site_xpos[e.robots[0].eef_site_id["right"]]
    bp = e.sim.data.body_xpos[e.obj_body_id["bowl"]]
    return dict(veg_in_bowl=bool(OU.check_obj_in_receptacle(e, "vegetable", "bowl")), veg_released=bool(_released(e, "vegetable")),
                bowl_in_mw=bool(OU.obj_inside_of(e, "bowl", e.microwave)), grip_bowl_dist=round(float(np.linalg.norm(gp - bp)), 3),
                door_closed=bool(e.microwave.is_closed(env=e)), turned_on=bool(e.microwave.get_state()["turned_on"]))


def work(job):
    import atomic_eval  # noqa: F401  (registers the envs, as in the rollout)
    from atomic_eval import load_scene
    from atomic_steps import TASKS
    from analyze_stages import replay_env
    from collect import replay
    from imagine import py_state, set_py_state, progress, progress_ref
    from oracle import set_state
    task, scene, d, tag, t, k, L, row_ps, k_replay_row, rl = job
    out = []
    try:
        env = None
        for attempt in range(2):
            try:
                env = replay_env(task)
                load_scene(env, scene)
                break
            except Exception as e:
                if attempt or "initialize" not in repr(e):
                    raise
        steps = TASKS[task]
        ep = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
        t0 = time.time()
        S0, k_rep = replay(env, ep, t, steps)
        ps = py_state(env)
        ps_diff = {n: {a: (v, row_ps.get(n, {}).get(a)) for a, v in dd.items() if row_ps.get(n, {}).get(a) != v
                       and not (isinstance(v, float) and abs(v - (row_ps.get(n, {}).get(a) or 0)) < 1e-9)}
                   for n, dd in ps.items()}
        ps_diff = {n: v for n, v in ps_diff.items() if v}
        ref = progress_ref(task, k, env)
        for c, m, lab in rl:
            z = np.load(f"{R}/runs/20261026_critic_data/{d}/roll/{tag}_t{t}_c{c}_m{m}.npz")
            st = z["states"]
            set_py_state(env, row_ps)
            set_state(env, st[0])
            start_diff = float(np.abs(st[0] - ep[t]).max())
            S = copy.deepcopy(S0)
            kk, success, stop_at, hist = k, False, None, []
            cond_steps, cond_bound = [], []  # steps where stage k's check holds (copy of S), and boundaries
            succ_steps = []
            n = len(st) - 1
            for i in range(1, n + 1):
                set_state(env, st[i])
                if env._check_success():
                    success = True; succ_steps.append(i)
                if k < len(steps) and steps[k]["check"](env, copy.deepcopy(S)):
                    cond_steps.append(i)
                if i % 16 == 0 or i == n:
                    if k < len(steps) and steps[k]["check"](env, copy.deepcopy(S)):
                        cond_bound.append(i)
                    while kk < len(steps) and steps[kk]["check"](env, S):
                        hist.append((i, kk)); kk += 1
                    if success or kk >= k + 1:
                        stop_at = i
                        break
            # the env is now at the step where the re-evaluated rollout stops (or at the last saved state); the labels are
            # computed there, as collect.py computes them at the rollout's end
            end = stop_at if stop_at is not None else n
            done = bool(kk > k or success)
            prev_ok = all(steps[q]["check"](env, copy.deepcopy(S)) for q in range(k)) if done and k < len(steps) - 1 else True
            prog = round(progress(task, k, env, ref), 3)
            extra = {}
            if task == "SteamInMicrowave":
                extra = conj_sim(env)
            res = dict(task=task, dir=d, tag=tag, t=t, k=k, L=L, c=c, m=m, stored=lab,
                       re=dict(done=done, prev_ok=prev_ok, steps=end, success=bool(success), reached=int(kk), progress=prog),
                       stop_at=stop_at, n_saved=n, start_diff=start_diff, k_replay=k_rep, k_replay_row=k_replay_row,
                       ps_diff=ps_diff, hist=hist, cond_steps_first=cond_steps[:3], cond_steps_n=len(cond_steps),
                       cond_bound=cond_bound[:5], succ_steps=succ_steps[:5], final=extra)
            res["match"] = {f: res["re"][f] == lab[f] for f in ("done", "prev_ok", "steps", "success", "reached")}
            res["match"]["progress"] = abs(res["re"]["progress"] - lab["progress"]) <= 0.0015
            res["match"]["stop"] = stop_at == n or (stop_at is None and n == L)  # stopped where it did, or ran out its budget
            out.append(res)
        for r in out:
            r["replay_s"] = round(time.time() - t0, 1)
        return out
    except Exception as e:
        import traceback
        return [dict(task=task, tag=tag, t=t, dir=d, error=repr(e), tb=traceback.format_exc()[-1500:])]


def main():
    sample, outp, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    jobs = json.load(open(sample))
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            r = json.loads(l)
            if "error" not in r:
                done.add((r["dir"], r["tag"], r["t"]))
    jobs = [tuple(j) for j in jobs if (j[2], j[3], j[4]) not in done]
    jobs.sort(key=lambda j: j[0])  # group by task: one env per task per worker
    print(f"{len(jobs)} points to check, {workers} workers", flush=True)
    with multiprocessing.get_context("spawn").Pool(workers, maxtasksperchild=40) as pool, open(outp, "a") as f:
        for res in pool.imap_unordered(work, jobs, chunksize=1):
            for r in res:
                f.write(json.dumps(r, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
            f.flush()
            bad = [r for r in res if "error" in r or not all(r["match"].values())]
            print(f"{res[0]['tag']} t{res[0]['t']} {len(res)} rollouts, mismatches {len(bad)}", flush=True)


if __name__ == "__main__":
    main()
