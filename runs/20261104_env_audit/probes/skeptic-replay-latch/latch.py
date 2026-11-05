"""Skeptic check of 'collect.py replay() drops the _check_success latches'.
For each episode of runs/20261025_seen480 of TASK, compare at every chunk boundary b (multiple of 16):
  A: collect.replay semantics (set_state per step; stage checks at boundaries) -- run through collect.replay itself
     for selected t, and through an identical loop that records the attr at every boundary;
  B: live semantics (set_state per step, then _check_success() twice, as reward() and the gym wrapper do in env.step).
usage: python -I latch.py TASK attr [tags,...]"""
import sys, json, copy, glob
ROOT = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{ROOT}/runs/scripts"); sys.path.insert(0, f"{ROOT}/third_party/Isaac-GR00T")
sys.path.insert(0, f"{ROOT}/runs/20261026_critic_data"); sys.path.insert(0, f"{ROOT}/runs/20261026_imagine")
import numpy as np
import robosuite, robocasa  # noqa
from robosuite.controllers import load_composite_controller_config
from atomic_eval import load_scene
from atomic_steps import TASKS
from oracle import set_state
from collect import replay


def make_env(task):
    cc = load_composite_controller_config(controller=None, robot="PandaOmron")
    env = robosuite.make(env_name=task, robots="PandaOmron", controller_configs=cc,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=256, camera_heights=256, has_renderer=False, has_offscreen_renderer=False,
        ignore_done=True, use_object_obs=True, use_camera_obs=False, camera_depths=False, seed=None,
        obj_instance_split="target", generative_textures=None, randomize_cameras=False,
        layout_and_style_ids=list(zip(range(1, 11), range(1, 11))), layout_ids=None, style_ids=None, translucent_robot=False)
    env.reset()
    return env

task, attr = sys.argv[1], sys.argv[2]
RUN = f"{ROOT}/runs/20261025_seen480/run"
jobs = {j["tag"]: j for j in json.load(open(f"{RUN}/jobs.json"))}
eps = {json.loads(l)["tag"]: json.loads(l) for p in glob.glob(f"{RUN}/results_w*.jsonl") for l in open(p)
       if json.loads(l).get("task") == task}
tags = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(eps, key=lambda s: int(s.split("_s")[1].split("_")[0]))
steps = TASKS[task]
env = make_env(task)
for tag in tags:
    r = eps[tag]; states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]; T = len(states) - 1
    # A: collect.replay's loop, recording at each boundary
    load_scene(env, jobs[tag]["scene"]); S, k = {}, 0; A = {}; KA = {}
    for i in range(T + 1):
        set_state(env, states[i])
        if i and i % 16 == 0:
            while k < len(steps) and steps[k]["check"](env, S):
                k += 1
            A[i] = copy.deepcopy(getattr(env, attr)); KA[i] = k
    # B: live semantics
    load_scene(env, jobs[tag]["scene"]); B = {}; succB = {}
    for i in range(1, T + 1):
        set_state(env, states[i]); env._check_success(); s = env._check_success()
        if i % 16 == 0:
            B[i] = copy.deepcopy(getattr(env, attr)); succB[i] = bool(s)
    diff = [b for b in A if A[b] != B[b]]
    # decision points as make_jobs would pick them (every 160, k < n_stages), with the main run's stage k
    dps = [t for t in range(0, r["steps"], 160) if sum(tk <= t for tk, _ in r["history"]) < len(steps)]
    dp_diff = [(t, sum(tk <= t for tk, _ in r["history"]), A.get(t), B.get(t)) for t in dps if t and A.get(t) != B.get(t)]
    # sanity: collect.replay itself at one differing boundary gives A's value
    chk = None
    if diff:
        load_scene(env, jobs[tag]["scene"]); replay(env, states, diff[0], steps)
        chk = copy.deepcopy(getattr(env, attr)) == A[diff[0]]
    first_succ_live = next((b for b in sorted(succB) if succB[b]), None)
    print("RESULT", json.dumps(dict(tag=tag, T=T, main_success=r["success"], history=r["history"], n_bound=len(A),
          n_diff=len(diff), first_diff=diff[:1], A_first=A[diff[0]] if diff else None, B_first=B[diff[0]] if diff else None,
          replay_fn_agrees=chk, dp_diff=dp_diff, first_live_success_boundary=first_succ_live,
          final_A=A[max(A)], final_B=B[max(B)]), default=str), flush=True)
