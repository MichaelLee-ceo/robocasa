"""Does the stale-kinematics mechanism explain live-vs-replay differences in seen480?

For each episode, replay its saved per-step states two ways:
  replay : what stage_trace / replay_check do (set_state = set + forward + update_sites + update_state, then _check_success)
  live   : emulate what env.step leaves behind: qpos/qvel = the saved (post-integration) values, but kinematics and
           contacts computed at the PRE-integration qpos of the last substep (qpos_old = integratePos(qpos, qvel, -dt),
           exact for Euler / implicitfast position integration), then _check_success, update_state, _check_success
           (reward = second call, as RoboCasaGymEnv.step).
Stage checks are evaluated in task order at chunk boundaries (every 16 steps and at the last step) with one S, as in
atomic_eval.rollout. Compare both with the live history / success / step count in results_w*.jsonl.
usage: python -I emulate_live.py tag [tag ...]"""
import sys, json, glob, os
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts"); sys.path.insert(0, f"{R}/runs/20261025_seen480")
import mujoco
from paths import local
from analyze_stages import replay_env
import atomic_eval  # noqa
from oracle import reset_to, set_state
from atomic_steps import TASKS

RUN = f"{R}/runs/20261025_seen480/run"
res = {}
for p in glob.glob(f"{RUN}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            res[r["tag"]] = r


def run(env, task, states, mode):
    m, d = env.sim.model._model, env.sim.data._data
    nq, nv = m.nq, m.nv
    steps, S, k, hist = TASKS[task], {}, 0, []
    first_reward = None
    T = len(states) - 1
    for f in range(len(states)):
        if mode == "replay" or f == 0:
            set_state(env, states[f])
            rew = env._check_success()
        else:
            env.sim.set_state_from_flattened(states[f])
            qpos_new = d.qpos.copy(); qvel_new = d.qvel.copy()
            qold = qpos_new.copy()
            mujoco.mj_integratePos(m, qold, qvel_new, -m.opt.timestep)
            d.qpos[:] = qold
            mujoco.mj_forward(m, d)
            d.qpos[:] = qpos_new; d.qvel[:] = qvel_new  # mjData qpos/qvel = post-integration, derived = pre
            env._check_success()       # reward() inside _post_action
            env.update_state()         # kitchen._post_action
            rew = env._check_success() # RoboCasaGymEnv.step
        if f > 0 and rew and first_reward is None:
            first_reward = f
        if f > 0 and (f % 16 == 0 or f == T):
            while k < len(steps) and steps[k]["check"](env, S):
                hist.append([f, k]); k += 1
            if first_reward is not None:  # rollout stops after the chunk in which a reward was seen
                return dict(hist=hist, success=True, steps=f, first_reward=first_reward)
    return dict(hist=hist, success=False, steps=T, first_reward=None)


if __name__ == "__main__":
    out = []
    for tag in sys.argv[1:]:
        r = res[tag]; task = r["task"]
        z = np.load(local(r["scene"]), allow_pickle=True)
        xml, ep_meta = str(z["xml"]), json.loads(str(z["ep_meta"]))
        states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
        env = replay_env(task)
        row = dict(tag=tag, live=dict(hist=r["history"], success=r["success"], steps=r["steps"]))
        for mode in ("replay", "live"):
            reset_to(env, xml, ep_meta)
            row[mode] = run(env, task, states, mode)
        row["emul_matches_live"] = (row["live"]["hist"] == r["history"] and row["live"]["success"] == r["success"]
                                    and row["live"]["steps"] == r["steps"])
        row["replay_matches_live"] = (row["replay"]["hist"] == r["history"] and row["replay"]["success"] == r["success"]
                                      and row["replay"]["steps"] == r["steps"])
        print(json.dumps(row), flush=True)
