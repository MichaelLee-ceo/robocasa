import os
import sys
sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scripts"); sys.path.insert(0, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/third_party/Isaac-GR00T")
import numpy as np, gymnasium as gym, robocasa, groot_eval
from atomic_eval import load_scene
def mk(): b = gym.make("robocasa/StackBowlsCabinet", split="target", enable_render=False).unwrapped; return b, b.env
b1, k1 = mk(); b1.reset(seed=0)
b2, k2 = mk(); load_scene(k2, os.path.expanduser(os.environ.get("ROBOCASA_ROOT", "~/Desktop/robocasa")) + "/runs/scenes_seen5/StackBowlsCabinet_s0.npz")
m1, m2 = k1.sim.model._model, k2.sim.model._model
o1, o2 = m1.opt, m2.opt
for f in ("timestep", "integrator", "cone", "impratio", "noslip_iterations", "iterations", "tolerance", "jacobian", "solver", "o_margin"):
    try: print("opt", f, getattr(o1, f), getattr(o2, f))
    except Exception as e: pass
print("opt.enableflags/disableflags", o1.enableflags, o2.enableflags, o1.disableflags, o2.disableflags)
for f in ("body_mass", "body_inertia", "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_margin", "geom_gap", "geom_priority", "geom_contype", "geom_conaffinity",
          "dof_damping", "dof_armature", "dof_frictionloss", "jnt_stiffness", "actuator_gainprm", "actuator_biasprm", "body_gravcomp", "geom_size", "geom_type", "mesh_vertnum"):
    a, b = np.asarray(getattr(m1, f)), np.asarray(getattr(m2, f))
    if a.shape != b.shape: print(f, "shape", a.shape, b.shape); continue
    d = np.abs(a.astype(float) - b.astype(float)); 
    if d.max() > 1e-9:
        idx = np.unravel_index(d.argmax(), d.shape)
        print(f, "maxdiff", float(d.max()), "count", int((d.reshape(len(d), -1).max(1) > 1e-9).sum()), "at", idx, a[idx[0]], b[idx[0]])
for n in ("bowl1", "bowl2"):
    bid1 = k1.obj_body_id[n]; bid2 = k2.obj_body_id[n]
    print(n, "mass", m1.body_mass[bid1], m2.body_mass[bid2])
