"""At split-hold boundaries of StackBowlsCabinet main runs: restore with set_state + settle (as collect.py) vs. the
same with the gripper ramp set to +1, then hold the arm still and command close for 16 steps. Bowl displacement.
usage: python -I drop_test.py s:t[,s:t...]"""
import json, sys, glob
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite
from robosuite.controllers import load_composite_controller_config
from oracle import reset_to, set_state
import hidden_state as H

task = "StackBowlsCabinet"
env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
                     has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, use_object_obs=True,
                     obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), translucent_robot=False)
for st in sys.argv[1].split(","):
    s, t = map(int, st.split(":"))
    z = np.load(glob.glob(f"{R}/runs/scenes_seen*_v2*/{task}_s{s}.npz")[0], allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"])
    ep = np.load(f"{R}/runs/20261025_seen480/run/{task}/{task}_s{s}_single_steps.npz")["states"]
    cc = env.robots[0].composite_controller
    gi = cc._action_split_indexes["right_gripper"]
    print({k: v for k, v in cc._action_split_indexes.items()}, env.action_dim)
    bids = {n: env.obj_body_id[n] for n in env.objects if n.startswith("bowl")}
    set_state(env, ep[t]); H.settle(env, ep[t - 1]); h = H.capture(env)
    ramp0 = float(h["robots"][0]["grippers"]["right"]["current_action"][1])
    res = {}
    for var in ("settle", "ramp+1"):
        set_state(env, ep[t]); H.restore(env, h)
        if var == "ramp+1":
            env.robots[0].gripper["right"].current_action = np.array([-1.0, 1.0])
        p0 = {n: np.array(env.sim.data.body_xpos[b]) for n, b in bids.items()}
        a = np.zeros(env.action_dim); a[gi[0]:gi[1]] = 1.0; a[-1] = -1.0
        gaps = []
        for k in range(16):
            env.step(a)
            q = env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes["right"]]; gaps.append(1000 * abs(q[0] - q[1]))
        res[var] = dict(dz_mm={n: round(1000 * float(env.sim.data.body_xpos[b][2] - p0[n][2]), 1) for n, b in bids.items()},
                        gap_mm=[round(gaps[i], 1) for i in (0, 3, 7, 15)])
    z0 = {n: round(float(ep[t][0] * 0), 1) for n in bids}
    print(f"s{s} t{t} ramp_after_settle={ramp0:+.1f} bowl z={ {n: round(float(np.array(env.sim.data.body_xpos[b])[2]), 3) for n, b in bids.items()} }")
    for var, r in res.items():
        print(f"   {var:7s} bowl dz after 16 steps {r['dz_mm']}  gap at steps 1/4/8/16 {r['gap_mm']}")
