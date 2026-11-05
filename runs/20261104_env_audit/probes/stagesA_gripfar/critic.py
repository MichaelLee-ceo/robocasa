"""Critic data, SteamInMicrowave: at the last saved state of every k=1 and k=2 rollout (the finishing boundary if done),
evaluate the stage-0 / stage-1 conjuncts and the door."""
import sys, json, os, pickle
import numpy as np
from multiprocessing import Pool
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plain import make_env
D = f"{R}/runs/20261026_critic_data/data/SteamInMicrowave"

def work(args):
    tag, items = args
    from atomic_eval import load_scene
    from oracle import set_state
    from atomic_steps import _released
    import robocasa.utils.object_utils as OU
    from paths import local
    s = tag.split("_")[1]
    sc = f"{R}/runs/scenes_seen5_v2/SteamInMicrowave_{s}.npz"
    if not os.path.exists(sc):
        sc = f"{R}/runs/scenes_seen5_v2_s30/SteamInMicrowave_{s}.npz"
    env = make_env("SteamInMicrowave")
    load_scene(env, sc)
    mw = env.microwave
    jadr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
    res = []
    for (d, t, k, c, m, done, prev_ok, success, steps) in items:
        st = np.load(f"{D}/{d}/roll/{tag}_t{t}_c{c}_m{m}.npz")["states"]
        door0 = None
        set_state(env, st[0]); door0 = float(env.sim.data.qpos[jadr])
        set_state(env, st[-1])
        eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        dist = float(np.linalg.norm(eef - env.sim.data.body_xpos[env.obj_body_id["bowl"]]))
        res.append(dict(d=d, tag=tag, t=t, k=k, c=c, m=m, done=done, prev_ok=prev_ok, success=success, steps=steps,
                        vib=bool(OU.check_obj_in_receptacle(env, "vegetable", "bowl")), relveg=bool(_released(env, "vegetable")),
                        bim=bool(OU.obj_inside_of(env, "bowl", env.microwave)), dist=dist, relbowl=bool(_released(env, "bowl")),
                        door0=door0, door=float(env.sim.data.qpos[jadr]), closed=bool(mw.is_closed(env=env))))
    env.close()
    return res

if __name__ == "__main__":
    by = {}
    for d in ["coarse", "dense", "confirm", "control"]:
        for l in open(f"{D}/{d}/results.jsonl"):
            r = json.loads(l)
            if "error" in r or r["k"] not in (1, 2):
                continue
            for cd in r["cands"]:
                for ro in cd["rollouts"]:
                    by.setdefault(r["tag"], []).append((d, r["t"], r["k"], cd["cand"], ro["m"], ro["done"], ro["prev_ok"], ro["success"], ro["steps"]))
    out = []
    with Pool(20) as p:
        for res in p.imap_unordered(work, sorted(by.items())):
            out += res
            print(len(out), flush=True)
    pickle.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "critic.pkl"), "wb"))
