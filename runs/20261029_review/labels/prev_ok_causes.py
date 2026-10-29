"""Task 3: why prev_ok is False. Every current SteamInMicrowave rollout with done and not prev_ok (the only task with
any): the conjuncts of the earlier stages' checks at the rollout's end state (where collect.py evaluates prev_ok).
S0 = vegetable in the bowl (contact + xy) and not touched by the gripper; S1 = bowl inside the microwave and gripper
> 0.25 m from the bowl. A conjunct that is a persistent fact of the world (vegetable in bowl, bowl inside) means the
earlier stage was really undone; the gripper conjuncts (released, far) are transient.
Writes prev_ok_causes.json. usage: python prev_ok_causes.py"""
import collections, json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261028_critical", f"{R}/runs/20261026_imagine",
                f"{R}/runs/20261026_critic_data", os.path.dirname(os.path.abspath(__file__))]


def main():
    import atomic_eval  # noqa: F401
    from atomic_eval import load_scene
    from analyze_stages import replay_env
    from common import HERE, coarse, dense, rolls, roll_path, scenes
    from imagine import set_py_state
    from oracle import set_state
    from label_check import conj_sim
    task = "SteamInMicrowave"
    sc = scenes()
    rows = list(coarse(task).values()) + list(dense(task).values())
    todo = [(r, c, m, x) for r in rows for c, m, x in rolls(r) if x["done"] and not x["prev_ok"]]
    print(f"{len(todo)} done rollouts with prev_ok False", flush=True)
    env, cur, out = replay_env(task), None, []
    for r, c, m, x in sorted(todo, key=lambda z: z[0]["tag"]):
        if cur != r["tag"]:
            load_scene(env, sc[r["tag"]]); cur = r["tag"]
        st = np.load(roll_path(r, c, m))["states"]
        set_py_state(env, r["py_state"])
        set_state(env, st[-1])
        f = conj_sim(env)
        s0 = f["veg_in_bowl"] and f["veg_released"]
        s1 = f["bowl_in_mw"] and f["grip_bowl_dist"] > 0.25
        cause = []
        if not f["veg_in_bowl"]:
            cause.append("vegetable out of bowl")
        elif not f["veg_released"]:
            cause.append("gripper touching vegetable")
        if r["k"] >= 2:
            if not f["bowl_in_mw"]:
                cause.append("bowl not inside")
            elif f["grip_bowl_dist"] <= 0.25:
                cause.append("gripper within 0.25 m of bowl")
        out.append(dict(tag=r["tag"], t=r["t"], k=r["k"], c=c, m=m, dir=r["_dir"], steps=x["steps"], s0=s0, s1=s1, cause=cause, **f))
    json.dump(out, open(os.path.join(HERE, "prev_ok_causes.json"), "w"), indent=0)
    for k in (1, 2):
        g = [o for o in out if o["k"] == k]
        print(f"S{k}: {len(g)} rollouts; causes:", dict(collections.Counter(" + ".join(o["cause"]) or "none (re-eval says prev_ok True)" for o in g)))
        pts = collections.Counter((o["tag"], o["t"]) for o in g)
        print("   points:", len(pts), "most:", pts.most_common(6))


if __name__ == "__main__":
    main()
