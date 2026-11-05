"""Independent check of the stale-kinematics mechanism.
After env.step: are live xpos/contacts == kinematics of qpos_old = integratePos(qpos, qvel, -dt) (start of last substep)?
Same with lite_physics=False (plain mj_forward + mj_step loop)?"""
import sys, json
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-robot-sim")
from harness import *
import mujoco
from atomic_eval import load_scene
task = sys.argv[1]; lite = sys.argv[2] == "1"
env = make_env(task)
load_scene(env, scene_path(task, 0))
env.lite_physics = lite
m, d = env.sim.model._model, env.sim.data._data
print("integrator", m.opt.integrator, "dt", m.opt.timestep, "substeps", int(env.control_timestep / env.model_timestep))
d2 = mujoco.MjData(m)
def kin(qpos):
    d2.qpos[:] = qpos; d2.qvel[:] = d.qvel; d2.act[:] = d.act; d2.time = d.time
    mujoco.mj_fwdPosition(m, d2)
    return d2.xpos.copy(), {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d2.contact[:d2.ncon]}
acts = rand_actions(160, 7)
e_old, e_new, cdiff_old, cdiff_new = [], [], 0, 0
for a in acts:
    env.step(action_vec(env, *a))
    live = d.xpos.copy(); lc = {(min(c.geom1, c.geom2), max(c.geom1, c.geom2)) for c in d.contact[:d.ncon]}
    qold = d.qpos.copy(); mujoco.mj_integratePos(m, qold, d.qvel, -m.opt.timestep)
    xo, co = kin(qold); xn, cn = kin(d.qpos.copy())
    e_old.append(np.abs(live - xo).max()); e_new.append(np.abs(live - xn).max())
    cdiff_old += lc != co; cdiff_new += lc != cn
print("RESULT", json.dumps(dict(task=task, lite=lite, steps=len(acts),
    max_abs_live_vs_preintegration_m=float(np.max(e_old)), max_abs_live_vs_fresh_m=float(np.max(e_new)),
    mean_abs_live_vs_fresh_mm=round(1000*float(np.mean(e_new)),3),
    contactset_diff_vs_preint=int(cdiff_old), contactset_diff_vs_fresh=int(cdiff_new))))
