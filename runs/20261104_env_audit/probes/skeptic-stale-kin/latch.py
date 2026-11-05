"""Real-policy episodes of seen480: per saved step, the contact-gated latch inputs and the official success under
 live  : what env.step leaves (qpos/qvel post-integration, kinematics + contacts at the last substep's start =
         integratePos(qpos, qvel, -dt)), then update_state, then _check_success (RoboCasaGymEnv.step's reward)
 replay: set_state (set + forward + update_sites + update_state), then _check_success  (stage_trace / collect.replay)
Validates 'live' against the logged live outcome (success, steps). usage: latch.py tag [tag...]"""
import sys, json, glob
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
        if "error" not in r: res[r["tag"]] = r

def button_contacts(env, task):
    g = env.robots[0].gripper["right"]
    if task == "SteamInMicrowave":
        mw = env.microwave
        return (bool(env.check_contact(g, f"{mw.name}_start_button")), bool(env.check_contact(g, f"{mw.name}_stop_button")))
    cm = env.coffee_machine
    return (any(env.check_contact(g, f"{cm.naming_prefix}{b}") for b in cm._start_button_names), False)

def latch(env, task):
    return bool(env.microwave._turned_on if task == "SteamInMicrowave" else env.coffee_machine._turned_on)

def run(env, task, states, mode):
    m, d = env.sim.model._model, env.sim.data._data
    out = dict(contact=[], latch=[], succ=[], stage=[])
    steps, S, k = TASKS[task], {}, 0
    for f in range(len(states)):
        if mode == "replay" or f == 0:
            set_state(env, states[f])
        else:
            env.sim.set_state_from_flattened(states[f])
            qn, vn = d.qpos.copy(), d.qvel.copy()
            qo = qn.copy(); mujoco.mj_integratePos(m, qo, vn, -m.opt.timestep)
            d.qpos[:] = qo; mujoco.mj_forward(m, d); d.qpos[:] = qn; d.qvel[:] = vn
            env._check_success(); env.update_state()
        out["contact"].append(button_contacts(env, task)); out["latch"].append(latch(env, task))
        out["succ"].append(bool(env._check_success()))
        if f > 0 and (f % 16 == 0 or f == len(states) - 1):
            while k < len(steps) and steps[k]["check"](env, S):
                k += 1
            out["stage"].append(k)
    return out

for tag in sys.argv[1:]:
    r = res[tag]; task = r["task"]
    z = np.load(local(r["scene"]), allow_pickle=True)
    xml, ep_meta = str(z["xml"]), json.loads(str(z["ep_meta"]))
    states = np.load(f"{RUN}/{task}/{tag}_steps.npz")["states"]
    env = replay_env(task)
    o = {}
    for mode in ("live", "replay"):
        reset_to(env, xml, ep_meta); o[mode] = run(env, task, states, mode)
    L, P = o["live"], o["replay"]
    first = lambda xs: next((i for i, x in enumerate(xs) if x and i > 0), None)
    fl, fp = first(L["succ"]), first(P["succ"])
    live_steps_emul = (min(len(states) - 1, ((fl + 15) // 16) * 16) if fl is not None else len(states) - 1)
    row = dict(tag=tag, task=task, logged=dict(success=r["success"], steps=r["steps"]),
               emul_live_ok=(fl is not None) == r["success"] and live_steps_emul == r["steps"],
               n=len(states), contact_frames_live=sum(any(c) for c in L["contact"]), contact_frames_replay=sum(any(c) for c in P["contact"]),
               contact_diff_frames=sum(a != b for a, b in zip(L["contact"], P["contact"])),
               latch_first_live=first(L["latch"]), latch_first_replay=first(P["latch"]),
               latch_diff_frames=sum(a != b for a, b in zip(L["latch"], P["latch"])),
               succ_first_live=fl, succ_first_replay=fp, succ_diff_frames=sum(a != b for a, b in zip(L["succ"], P["succ"])),
               stage_diff_boundaries=sum(a != b for a, b in zip(L["stage"], P["stage"])), final_stage=(L["stage"][-1], P["stage"][-1]))
    print("ROW", json.dumps(row), flush=True)
