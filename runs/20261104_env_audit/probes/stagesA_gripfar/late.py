"""Per-step replay of the k=1 critic rollouts whose stage-1 credit came after the door had moved."""
import sys, os, pickle
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
    s = tag.split("_")[1]
    sc = f"{R}/runs/scenes_seen5_v2/SteamInMicrowave_{s}.npz"
    if not os.path.exists(sc):
        sc = f"{R}/runs/scenes_seen5_v2_s30/SteamInMicrowave_{s}.npz"
    env = make_env("SteamInMicrowave"); load_scene(env, sc)
    mw = env.microwave; jadr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
    res = []
    for r in items:
        st = np.load(f"{D}/{r['d']}/roll/{tag}_t{r['t']}_c{r['c']}_m{r['m']}.npz")["states"]
        rows = []
        for f in range(len(st)):
            set_state(env, st[f])
            eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
            rows.append((f, bool(OU.obj_inside_of(env, "bowl", mw)), bool(_released(env, "bowl")),
                         float(np.linalg.norm(eef - env.sim.data.body_xpos[env.obj_body_id["bowl"]])), float(env.sim.data.qpos[jadr]),
                         bool(OU.check_obj_in_receptacle(env, "vegetable", "bowl"))))
        res.append((r, rows))
    env.close()
    return res

if __name__ == "__main__":
    out = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "critic.pkl"), "rb"))
    sel = [r for r in out if r["k"] == 1 and ((r["done"] and r["door"] - r["door0"] > 0.2) or (not r["done"] and r["bim"] and r["relbowl"]))]
    by = {}
    for r in sel: by.setdefault(r["tag"], []).append(r)
    allres = []
    with Pool(12) as p:
        for res in p.imap_unordered(work, sorted(by.items())):
            allres += res
    pickle.dump(allres, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "late.pkl"), "wb"))
    for r, rows in allres:
        bim = np.array([x[1] for x in rows]); rel = np.array([x[2] for x in rows]); d = np.array([x[3] for x in rows]); door = np.array([x[4] for x in rows])
        i = np.nonzero(bim & rel)[0]; t_rel = int(i[0]) if len(i) else None
        i = np.nonzero(door - door[0] > 0.2)[0]; t_door = int(i[0]) if len(i) else None
        i = np.nonzero(bim & (d > 0.25))[0]; t_far = int(i[0]) if len(i) else None
        far_between = None
        if t_rel is not None and t_door is not None and t_door > t_rel:
            far_between = bool((d[t_rel:t_door] > 0.25).any())
        bounds = [f for f in range(16, len(rows), 16)]
        # first chunk boundary with bowl in mw & released
        b_rel = next((f for f in bounds if bim[f] and rel[f]), None)
        print(r["d"], r["tag"].split("_")[1], "t", r["t"], "c", r["c"], "m", r["m"], "done", r["done"], "steps", r["steps"],
              "| first step in+released", t_rel, "door starts", t_door, "first in+far step", t_far, "| far>25cm between release and door:", far_between,
              "| min d after release %.3f" % (d[t_rel:].min() if t_rel is not None else -1), "first boundary in+released", b_rel, flush=True)
