"""Task 3, PackIdenticalLunches: the stage checks of S0-S2 only count objects in any tupperware, while S3 is the official
success (one vegetable and one meat in each tupperware, nothing doubled, gripper far). At every current point with
k >= 1, which tupperware holds which object in the main run at t, and whether the official success is still reachable
by adding the missing objects without moving a placed one (else V of S3, and the meaning of S1/S2 completions, depend
on undoing earlier placements). Writes pil_assign.json. usage: python pil_assign.py"""
import json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261028_critical", f"{R}/runs/20261026_imagine",
                f"{R}/runs/20261026_critic_data", os.path.dirname(os.path.abspath(__file__))]
OBJS = ["vegetable0", "vegetable1", "meat0", "meat1"]


def reachable(where):
    """where: obj -> list of tupperwares holding it. Success reachable by only adding objects?"""
    cnt = {"tupperware0": {"veg": 0, "meat": 0}, "tupperware1": {"veg": 0, "meat": 0}}
    for o, ws in where.items():
        if len(ws) > 1:
            return False
        for w in ws:
            cnt[w]["veg" if o.startswith("veg") else "meat"] += 1
    return all(c["veg"] <= 1 and c["meat"] <= 1 for c in cnt.values())


def main():
    import atomic_eval  # noqa: F401
    from atomic_eval import load_scene
    from analyze_stages import replay_env
    import robocasa.utils.object_utils as OU
    from common import HERE, SEEN, coarse, dense, rolls, scenes
    from oracle import set_state
    task = "PackIdenticalLunches"
    sc = scenes()
    pts = {**{k: ("coarse", r) for k, r in coarse(task).items()}, **{k: ("dense", r) for k, r in dense(task).items()}}
    env, cur, out = replay_env(task), None, []
    for (tag, t), (src, r) in sorted(pts.items()):
        if r["k"] < 1:
            continue
        if cur != tag:
            load_scene(env, sc[tag]); cur = tag
            ms = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
        set_state(env, ms[t])
        where = {o: [w for w in ("tupperware0", "tupperware1") if OU.check_obj_in_receptacle(env, o, w)] for o in OBJS}
        xs = [x for _, _, x in rolls(r)]
        out.append(dict(tag=tag, t=t, k=r["k"], src=src, where=where, placed=sum(bool(w) for w in where.values()),
                        reachable=reachable(where), V=float(np.mean([x["done"] for x in xs]))))
        o = out[-1]
        print(f"{tag:32s} t{t:5d} S{o['k']} placed {o['placed']} reachable {o['reachable']} V {o['V']:.2f} {where}", flush=True)
    json.dump(out, open(os.path.join(HERE, "pil_assign.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
