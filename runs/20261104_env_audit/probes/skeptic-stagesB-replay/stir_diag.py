"""StirVegetables s7: per-frame conjuncts of the official stir counter around the end, in plain replay and with
kinematics moved back one substep (stale emulation). usage: python -I stir_diag.py <seed> <from>"""
import glob, json, sys
import numpy as np
import mujoco
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
import robocasa.utils.object_utils as OU  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa

task = "StirVegetables"; seed = int(sys.argv[1]); f0 = int(sys.argv[2])
rows = [json.loads(l) for p in glob.glob(f"{R}/runs/20261025_seen480/run/results_w*.jsonl") for l in open(p)]
r = [x for x in rows if x["task"] == task and x["seed"] == seed and "error" not in x][0]
mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name=task, split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
z = np.load(local(r["scene"]), allow_pickle=True)
states = np.load(f"{R}/runs/20261025_seen480/run/{task}/{r['tag']}_steps.npz")["states"]
reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
m, d = env.sim.model._model, env.sim.data._data
prev = {}


def conj(e):
    sp = e.objects["spatula"]
    out = dict(grasp=bool(OU.check_obj_grasped(e, "spatula")),
               pot=bool(e.stove.check_obj_location_on_stove(env=e, obj_name="pot", threshold=0.15) == e.knob))
    for o in ("veg1", "veg2"):
        p = np.array(e.sim.data.body_xpos[e.obj_body_id[e.objects[o].name]])
        out[o] = dict(inpot=bool(OU.check_obj_in_receptacle(e, o, "pot")), touch=bool(e.check_contact(sp, e.objects[o])), pos=p)
    return out


rows_out = []
for f in range(len(states)):
    set_state(env, states[f])
    if f < f0:
        continue
    a = conj(env)
    st = env.sim.get_state().flatten()
    q = np.array(d.qpos); v = np.array(d.qvel)
    mujoco.mj_integratePos(m, q, v, -m.opt.timestep)
    dq = float(np.abs(q - d.qpos).max())
    d.qpos[:] = q; env.sim.forward()
    b = conj(env)
    env.sim.set_state_from_flattened(st); env.sim.forward()
    line = dict(f=f, succ_time=int(env.success_time), dq=round(dq, 6))
    for k, c in (("rep", a), ("stale", b)):
        mv = {}
        for o in ("veg1", "veg2"):
            pp = prev.get((k, o))
            mv[o] = None if pp is None else round(float(np.linalg.norm((c[o]["pos"] - pp)[:2]) * 1e2), 3)
            if c["grasp"]:
                prev[(k, o)] = c[o]["pos"]
        line[k] = dict(g=int(c["grasp"]), pot=int(c["pot"]), v1=[int(c["veg1"]["inpot"]), int(c["veg1"]["touch"]), mv["veg1"]],
                       v2=[int(c["veg2"]["inpot"]), int(c["veg2"]["touch"]), mv["veg2"]])
    print(json.dumps(line), flush=True)
