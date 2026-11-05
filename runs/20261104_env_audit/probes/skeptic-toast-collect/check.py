"""Skeptic check of 'collect.py replay() leaves env.toaster_on False for GetToastedBread decision points after pop-up'.
Uses collect.py's own replay() and atomic_eval.load_scene (the code path decision_point() runs), on decision points that
make_jobs.py would generate (t multiple of 160 with EVERY=160), at the first such t with k=1.
For each point: (a) collect-style replay -> env.toaster_on, k, S; (b) same replay but env._check_success() per step
(what the live main run maintained); (c) oracle: put the bread on the plate (free joint teleport), step 10 zero actions
via env.step (reward() -> _check_success runs live), report official success with toaster_on as left by (a), then the
same with env.toaster_on forced True.
usage: python -I check.py seed [seed ...]"""
import glob, json, sys, os
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts"); sys.path.insert(0, f"{R}/runs/20261026_critic_data"); sys.path.insert(0, f"{R}/runs/20261026_imagine")
import gymnasium as gym
import collect  # the real module
from atomic_eval import load_scene
from oracle import set_state
from atomic_steps import TASKS
import robocasa.utils.object_utils as OU

task = "GetToastedBread"; steps = TASKS[task]
RUN = f"{R}/runs/20261025_seen480/run"
rows = [json.loads(l) for p in glob.glob(f"{RUN}/results_w*.jsonl") for l in open(p)]
scene = {j["tag"]: j["scene"] for j in json.load(open(f"{RUN}/jobs.json"))}
base = gym.make(f"robocasa/{task}", split="target", enable_render=False).unwrapped
kenv = base.env

def teleport_and_step(kenv, force_on):
    if force_on is not None:
        kenv.toaster_on = force_on
    m, d = kenv.sim.model, kenv.sim.data
    obj = kenv.objects["obj"]; jn = obj.joints[0]
    adr = m.get_joint_qpos_addr(jn)[0]
    vadr = m.get_joint_qvel_addr(jn)[0]
    pp = np.array(d.body_xpos[kenv.obj_body_id["plate"]])
    d.qpos[adr:adr+3] = pp + np.array([0, 0, 0.04]); d.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    d.qvel[vadr:vadr+6] = 0
    kenv.sim.forward()
    rews = []
    for _ in range(15):
        _, r, _, _ = kenv.step(np.zeros(kenv.action_dim))
        rews.append(r)
    return dict(toaster_on=bool(kenv.toaster_on), in_plate=bool(OU.check_obj_in_receptacle(kenv, "obj", "plate")),
                far=bool(OU.gripper_obj_far(kenv, "obj")), success=bool(kenv._check_success()), any_reward=bool(np.any(np.array(rews) > 0)))

for seed in map(int, sys.argv[1:]):
    r = [x for x in rows if x.get("task") == task and x.get("seed") == seed and "error" not in x][0]
    tag = r["tag"]
    states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
    t_done = r["history"][0][0]
    ts = [t for t in range(0, r["steps"], 160) if sum(tk <= t for tk, _ in r["history"]) == 1]
    t = ts[0]
    out = dict(tag=tag, stage0_live_t=t_done, decision_t=t, n_states=len(states))
    load_scene(kenv, scene[tag])
    out["after_load_scene_toaster_on"] = bool(kenv.toaster_on)
    S, k = collect.replay(kenv, states, t, steps)
    out["collect_replay"] = dict(k=k, S=S, toaster_on=bool(kenv.toaster_on), fx=kenv.toaster.get_state(kenv))
    out["collect_oracle_place"] = teleport_and_step(kenv, None)
    # same replay but _check_success per step (as the live run's reward() did)
    load_scene(kenv, scene[tag])
    S2, k2 = {}, 0
    first_on = None
    for i in range(t + 1):
        set_state(kenv, states[i]); kenv._check_success()
        if first_on is None and kenv.toaster_on: first_on = i
        if i and i % 16 == 0:
            while k2 < len(steps) and steps[k2]["check"](kenv, S2): k2 += 1
    out["perstep_replay"] = dict(k=k2, toaster_on=bool(kenv.toaster_on), latched_at=first_on)
    # oracle with toaster_on forced True from the collect-style replay
    load_scene(kenv, scene[tag]); collect.replay(kenv, states, t, steps)
    out["collect_oracle_place_forced_on"] = teleport_and_step(kenv, True)
    print(json.dumps(out, default=str), flush=True)
