"""The rollouts whose stored prev_ok is False while the earlier-stage checks hold at their saved end state when replayed
(prev_ok_causes.json, cause == []). Prints the conjuncts at the last 6 saved states, to see whether a contact flickers
at the end (the online check runs on the contacts of the last substep before integration, set_state + forward on the
saved post-integration state). usage: python prev_ok_mismatch.py"""
import json, os, sys
import numpy as np

R = "/home/michaellee/mclee/robocasa"
sys.path[:0] = [f"{R}/runs/scripts", f"{R}/runs/20261025_seen480", f"{R}/runs/20261028_critical", f"{R}/runs/20261026_imagine",
                f"{R}/runs/20261026_critic_data", os.path.dirname(os.path.abspath(__file__))]


def main():
    import atomic_eval  # noqa: F401
    from atomic_eval import load_scene
    from analyze_stages import replay_env
    from common import HERE, DATA, scenes
    from imagine import set_py_state
    from label_check import conj_sim
    from oracle import set_state
    import robocasa.utils.object_utils as OU
    rows = [o for o in json.load(open(os.path.join(HERE, "prev_ok_causes.json"))) if not o["cause"]]
    env, cur = replay_env("SteamInMicrowave"), None
    for o in sorted(rows, key=lambda o: o["tag"]):
        if cur != o["tag"]:
            load_scene(env, scenes()[o["tag"]]); cur = o["tag"]
        st = np.load(f"{DATA}/{o['dir']}/roll/{o['tag']}_t{o['t']}_c{o['c']}_m{o['m']}.npz")["states"]
        print(f"== {o['tag']} t{o['t']} S{o['k']} c{o['c']}m{o['m']} ({o['dir']}), {len(st) - 1} steps")
        for i in range(max(1, len(st) - 6), len(st)):
            set_state(env, st[i])
            f = conj_sim(env)
            vb = env.sim.data.body_xpos[env.obj_body_id["vegetable"]]; bb = env.sim.data.body_xpos[env.obj_body_id["bowl"]]
            th = env.objects["bowl"].horizontal_radius * 0.7
            print(f"   step {i}: {json.dumps(f)}  veg-bowl contact {env.check_contact(env.objects['vegetable'], env.objects['bowl'])} "
                  f"xy {np.linalg.norm(vb[:2] - bb[:2]):.3f} < {th:.3f}, veg z - bowl z {vb[2] - bb[2]:+.3f}")


if __name__ == "__main__":
    main()
