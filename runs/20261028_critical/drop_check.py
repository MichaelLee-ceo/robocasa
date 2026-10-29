"""Does restoring a decision point while the robot holds an object drop it? (found at SteamInMicrowave s13 t640: the
main run holds the bowl at 1.44 m, while every rollout from the same state has it at 0.54 m after 8 steps and on the
floor soon after)

For every collected decision point where the main run holds an object at t (touching and grasped in its trace), the
first 24 steps of rollout c0 m0 are compared with the main run's next 24 steps: the held object's height at steps 8,
16, 24, and the finger width. A point counts as an immediate drop when the object is >= 10 cm lower than in the main run
at step 16. Then, for those points, all 16 rollouts are checked the same way.
The rollouts are those of the coarse row in use (data.py). Writes drop_check.jsonl to $CRIT_OUT.
usage: python -I drop_check.py <workers>"""
import gzip, json, multiprocessing, os, sys
import numpy as np

R = os.environ["ROBOCASA_ROOT"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", HERE]
import data  # noqa: E402
SEEN = f"{R}/runs/20261025_seen480"


def heights(env, states, body, steps=(0, 8, 16, 24)):
    from oracle import set_state
    out = []
    for f in steps:
        if f >= len(states):
            out.append(None); continue
        set_state(env, states[f])
        w = sum(abs(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(q)]) for q in env.robots[0].gripper["right"].joints)
        out.append((round(float(env.sim.data.body_xpos[body][2]), 3), round(float(w), 4)))
    return out


def work(job):
    import atomic_eval  # noqa: F401
    from oracle import reset_to
    from analyze_stages import replay_env
    p, roll = job
    try:
        z = np.load(p["scene"], allow_pickle=True)
        env = replay_env(p["task"])
        reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
        body = env.obj_body_id[p["obj"]]
        ms = np.load(f"{SEEN}/run/{p['task']}/{p['tag']}_steps.npz")["states"][p["t"]:]
        main = heights(env, ms, body)
        rolls = {}
        for i in range(8):
            for m in (0, 1):
                st = np.load(f"{roll}/{p['tag']}_t{p['t']}_c{i}_m{m}.npz")["states"]
                rolls[f"c{i}m{m}"] = heights(env, st, body)
                if i == 0 and m == 0 and main[2] and rolls["c0m0"][2] and main[2][0] - rolls["c0m0"][2][0] < 0.10:
                    break  # c0 m0 holds it like the main run: no need to check the rest
            else:
                continue
            break
        drop = {k: bool(main[2] and v[2] and main[2][0] - v[2][0] >= 0.10) for k, v in rolls.items()}
        return dict(task=p["task"], tag=p["tag"], t=p["t"], k=p["k"], obj=p["obj"], pd=p["pd"], main=main, rolls=rolls, drop=drop)
    except Exception as e:
        import traceback
        return dict(tag=p["tag"], t=p["t"], error=repr(e), tb=traceback.format_exc()[-1000:])


def jobs():
    from effect_by_event import points
    tr, out = {}, []
    for p in points():
        if p["tag"] not in tr:
            tr[p["tag"]] = json.load(gzip.open(f"{SEEN}/analysis/{p['task']}/{p['tag']}_trace.json.gz"))
        held = [o for o, r in tr[p["tag"]]["obj"].items() if r["touch"][p["t"]] and r["grasp"][p["t"]]]
        if held:
            out.append((dict(task=p["task"], tag=p["tag"], t=p["t"], k=p["k"], pd=p["pd"], obj=held[0], scene=data.scenes()[p["tag"]]),
                        p["roll"]))
    return out


if __name__ == "__main__":
    js = jobs()
    print(f"{len(js)} decision points where the main run holds an object", flush=True)
    with multiprocessing.get_context("spawn").Pool(int(sys.argv[1])) as pool, open(os.path.join(data.OUT, "drop_check.jsonl"), "w") as f:
        for r in pool.imap_unordered(work, js):
            f.write(json.dumps(r) + "\n"); f.flush()
