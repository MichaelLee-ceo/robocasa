"""Per-step replay of the plain SteamInMicrowave runs (seen480): when is the bowl in the microwave and released,
when does stage 1's check (in microwave + eef > 25 cm) first hold, and what does the door do in between."""
import sys, json, glob, os
import numpy as np
from multiprocessing import Pool
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")

def make_env(task):
    import robosuite, robocasa
    from robosuite.controllers import load_composite_controller_config
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cc, camera_names=["robot0_agentview_left",
        "robot0_agentview_right", "robot0_eye_in_hand"], camera_widths=128, camera_heights=128, has_renderer=False,
        has_offscreen_renderer=False, ignore_done=True, use_object_obs=True, use_camera_obs=False, seed=None,
        obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), layout_ids=None, style_ids=None,
        translucent_robot=False)
    env.reset()
    return env

def work(tag):
    import gymnasium as gym
    import atomic_eval  # registers envs
    from atomic_eval import load_scene
    from oracle import set_state
    from atomic_steps import TASKS, _released
    import robocasa.utils.object_utils as OU
    run = f"{R}/runs/20261025_seen480/run"
    task = "SteamInMicrowave"
    sc = f"{R}/runs/scenes_seen5_v2/{tag.split('_')[0]}_{tag.split('_')[1]}.npz"
    if not os.path.exists(sc):
        sc = f"{R}/runs/scenes_seen5_v2_s30/{tag.split('_')[0]}_{tag.split('_')[1]}.npz"
    env = make_env(task)
    load_scene(env, sc)
    states = np.load(f"{run}/{task}/{tag}_steps.npz")["states"]
    mw = env.microwave
    jadr = env.sim.model.get_joint_qpos_addr(mw.door_joint_names[0])
    steps = TASKS[task]; S = {}; k = 0; hist = []
    rows = []
    for f in range(len(states)):
        set_state(env, states[f])
        succ = env._check_success()
        vib = OU.check_obj_in_receptacle(env, "vegetable", "bowl")
        bim = OU.obj_inside_of(env, "bowl", env.microwave)
        eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        d = float(np.linalg.norm(eef - env.sim.data.body_xpos[env.obj_body_id["bowl"]]))
        rel = _released(env, "bowl")
        door = float(env.sim.data.qpos[jadr])
        closed = mw.is_closed(env=env)
        on = mw.get_state()["turned_on"]
        rows.append((f, vib, bim, d, rel, door, closed, on, succ))
        if f > 0 and (f % 16 == 0 or f == len(states) - 1):
            while k < len(steps) and steps[k]["check"](env, S):
                hist.append([f, k]); k += 1
    env.close()
    return tag, hist, rows

if __name__ == "__main__":
    tags = sorted(os.path.basename(p)[:-len("_steps.npz")] for p in glob.glob(f"{R}/runs/20261025_seen480/run/SteamInMicrowave/*_steps.npz"))
    out = {}
    with Pool(15) as p:
        for tag, hist, rows in p.imap_unordered(work, tags):
            out[tag] = dict(hist=hist, rows=rows)
            print(tag, "done", flush=True)
    import pickle; pickle.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plain.pkl"), "wb"))
