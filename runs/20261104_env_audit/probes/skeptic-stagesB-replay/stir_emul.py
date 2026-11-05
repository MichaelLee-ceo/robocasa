"""StirVegetables seen480: success_time from (a) plain replay (set_state -> forward -> update_state, as stage_trace),
(b) 'stale-emulated' replay: qpos moved back by one substep (mj_integratePos with -dt and the saved qvel; exact
inverse of MuJoCo's position update), forward, then update_state -- i.e. what env.step's update_state sees (kinematics
and contacts from the start of the last substep, no forward after the final sim step). Compare with the live outcome.
usage: python -I stir_emul.py <seeds comma-separated>"""
import glob, json, sys
import numpy as np
import mujoco
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa

task = "StirVegetables"
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
rows = {r["seed"]: r for r in rows if r["task"] == task and "error" not in r}
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
m, d = env.sim.model._model, env.sim.data._data
print("integrator", m.opt.integrator, "timestep", m.opt.timestep, "control_timestep", env.control_timestep, flush=True)
nq = m.nq


def run(states, mode, z):
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
    m, d = env.sim.model._model, env.sim.data._data  # reset_to rebuilds the sim: use the current model/data
    for v in ("veg1", "veg2"):
        if hasattr(env, f"prev_obj_pos_{v}"):
            delattr(env, f"prev_obj_pos_{v}")
    env.success_time = 0
    inc, first5 = [], None
    for f in range(len(states)):
        a = env.success_time
        if mode == "replay" or f == 0:
            set_state(env, states[f])
        else:
            env.sim.set_state_from_flattened(states[f])
            q = np.array(d.qpos); v = np.array(d.qvel)
            mujoco.mj_integratePos(m, q, v, -m.opt.timestep)
            d.qpos[:] = q
            assert d is env.sim.data._data
            env.sim.forward()
            env.update_state()
        if env.success_time > a:
            inc.append(f)
        if first5 is None and env.success_time >= 5:
            first5 = f
    return dict(success_time=int(env.success_time), first_ge5=first5, inc=inc)


for seed in map(int, sys.argv[1].split(",")):
    r = rows[seed]
    z = np.load(local(r["scene"]), allow_pickle=True)
    states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
    out = dict(tag=r["tag"], live_success=r["success"], live_steps=r["steps"], n_states=len(states))
    for mode in ("replay", "stale_emul"):
        out[mode] = run(states, mode, z)
    print(json.dumps(out), flush=True)
