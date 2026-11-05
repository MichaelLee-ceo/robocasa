"""For SteamInMicrowave critic-data rollouts that finished stage k in (1, 2): the official conjuncts at the start
state and at the finish state (last saved state = the chunk boundary where stage k's check first held)."""
import json, os, sys, multiprocessing as mp
import numpy as np

R = "/home/michaellee/mclee/robocasa/runs"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{R}/20261028_critical", f"{R}/scripts", OUT]


def work(args):
    tag, items = args
    import mkenv
    from atomic_eval import load_scene
    from oracle import set_state
    import robocasa.utils.object_utils as OU
    import data
    env = mkenv.make("SteamInMicrowave")
    load_scene(env, data.scenes()[tag])
    g = env.robots[0].gripper["right"]

    def conj(s):
        set_state(env, s)
        return dict(veg_in_bowl=bool(OU.check_obj_in_receptacle(env, "vegetable", "bowl")),
                    bowl_in_mw=bool(OU.obj_inside_of(env, "bowl", env.microwave)),
                    gfar_bowl=bool(OU.gripper_obj_far(env, "bowl")),
                    released_veg=bool(not env.check_contact(g, env.objects["vegetable"])),
                    door_closed=bool(env.microwave.is_closed(env=env)),
                    veg_z=float(env.sim.data.body_xpos[env.obj_body_id["vegetable"]][2]),
                    d_veg_bowl=float(np.linalg.norm(env.sim.data.body_xpos[env.obj_body_id["vegetable"]][:2]
                                                    - env.sim.data.body_xpos[env.obj_body_id["bowl"]][:2])))
    res = []
    for it in items:
        st = np.load(it["path"], allow_pickle=True)["states"]
        it = dict(it)
        it["start"] = conj(st[0]); it["end"] = conj(st[-1])
        res.append(it)
    env.close()
    return res


def main():
    import data
    groups = {}
    for ds in ("coarse", "dense", "confirm", "control"):
        for (tag, t), r in data.rows("SteamInMicrowave", ds).items():
            if r["k"] not in (1, 2):
                continue
            for c in r["cands"]:
                for x in c["rollouts"]:
                    if not x["done"]:
                        continue
                    groups.setdefault(tag, []).append(dict(ds=ds, tag=tag, t=t, k=r["k"], c=c["cand"], m=x["m"],
                                                           prev_ok=x["prev_ok"], success=x["success"],
                                                           path=f"{data.roll(r)}/{tag}_t{t}_c{c['cand']}_m{x['m']}.npz"))
    with mp.Pool(min(20, len(groups))) as p:
        out = [y for ys in p.map(work, sorted(groups.items())) for y in ys]
    with open(f"{OUT}/probe.jsonl", "w") as f:
        for y in out:
            f.write(json.dumps(y) + "\n")
    print(len(out), "rollouts")


if __name__ == "__main__":
    main()
