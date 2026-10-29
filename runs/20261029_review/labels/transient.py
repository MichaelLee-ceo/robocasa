"""Transient task success: rollouts whose env._check_success() held at some step (so success = done = True) but whose
stage check failed at the chunk boundary (reached == k). Replays the saved states of the given rollouts and prints,
around every step where success held, the conjuncts of the official success.
usage: python transient.py <dir> <tag> <t> [c m ...]   (default: every rollout of the point with success and reached == k)"""
import copy, json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261028_critical", f"{R}/runs/20261026_imagine",
                f"{R}/runs/20261026_critic_data", os.path.dirname(os.path.abspath(__file__))]


def conj(task, e):
    import robocasa.utils.object_utils as OU
    if task == "KettleBoiling":
        knobs = e.stove.get_knobs_state(env=e)
        pos = np.array(e.sim.data.body_xpos[e.obj_body_id[e.objects["obj"].name]])[:2]
        out = dict(on_stove=bool(OU.check_obj_fixture_contact(e, "obj", e.stove)), far=bool(OU.gripper_obj_far(e)))
        gp = e.sim.data.site_xpos[e.robots[0].eef_site_id["right"]]
        out["grip_kettle"] = round(float(np.linalg.norm(gp - e.sim.data.body_xpos[e.obj_body_id["obj"]])), 3)
        for loc, site in e.stove.burner_sites.items():
            if site is None:
                continue
            d = float(np.linalg.norm(np.array(e.sim.data.get_site_xpos(site.get("name")))[:2] - pos))
            if d < 0.3:
                out[loc] = dict(d=round(d, 3), on=bool(e.stove.is_burner_on(env=e, burner_loc=loc)) if loc in knobs else None,
                                knob=round(float(knobs.get(loc, np.nan)), 3))
        return out
    if task == "SteamInMicrowave":
        return dict(veg_in_bowl=bool(OU.check_obj_in_receptacle(e, "vegetable", "bowl")), bowl_in=bool(OU.obj_inside_of(e, "bowl", e.microwave)),
                    door_closed=bool(e.microwave.is_closed(env=e)), on=bool(e.microwave.get_state()["turned_on"]),
                    door=round(float(list(e.microwave.get_joint_state(e, e.microwave.door_joint_names).values())[0]), 4))
    if task == "PackIdenticalLunches":
        out = {}
        for o in ("vegetable0", "vegetable1", "meat0", "meat1"):
            out[o] = [w for w in ("tupperware0", "tupperware1") if OU.check_obj_in_receptacle(e, o, w)]
        out["far"] = all(OU.gripper_obj_far(e, o) for o in ("vegetable0", "vegetable1", "meat0", "meat1"))
        return out


def main():
    import atomic_eval  # noqa: F401
    from atomic_eval import load_scene
    from analyze_stages import replay_env
    from common import read_rows, rolls, scenes
    from imagine import set_py_state
    from oracle import set_state
    d, tag, t = sys.argv[1], sys.argv[2], int(sys.argv[3])
    row = next(r for r in read_rows(d) if r["tag"] == tag and r["t"] == t)
    task = row["task"]
    want = [(int(sys.argv[i]), int(sys.argv[i + 1])) for i in range(4, len(sys.argv), 2)]
    if not want:
        want = [(c, m) for c, m, x in rolls(row) if x["success"] and x["reached"] == row["k"]]
    env = replay_env(task)
    load_scene(env, scenes()[tag])
    for c, m in want:
        x = next(x for cc, mm, x in rolls(row) if (cc, mm) == (c, m))
        st = np.load(f"{R}/runs/20261026_critic_data/{d}/roll/{tag}_t{t}_c{c}_m{m}.npz")["states"]
        set_py_state(env, row["py_state"])
        set_state(env, st[0])
        succ = []
        for i in range(1, len(st)):
            set_state(env, st[i])
            if env._check_success():
                succ.append(i)
        print(f"== {tag} t{t} c{c} m{m} label {x}  success at steps {succ[:20]}{' ...' if len(succ) > 20 else ''} (of {len(st) - 1})")
        show = sorted({j for i in succ[:3] for j in (i - 1, i, i + 1)} | {len(st) - 1})
        set_py_state(env, row["py_state"]); set_state(env, st[0])
        for i in range(1, len(st)):
            set_state(env, st[i])
            if i in show:
                print(f"   step {i}: success={env._check_success()} {json.dumps(conj(task, env))}")


if __name__ == "__main__":
    main()
