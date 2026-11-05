import sys, time; sys.path.insert(0, sys.argv[1])
from harness import *
from atomic_eval import load_scene
t0=time.time()
env = make_env("WashLettuce"); print("make", time.time()-t0)
s = load_scene(env, scene_path("WashLettuce", 0)); print("load", time.time()-t0)
m = env.sim.model
print("nmocap", m.nmocap, "neq", m.neq, "nuserdata", m.nuserdata, "na", m.na, "nplugin", getattr(m,'nplugin',None), "integrator", m.opt.integrator, "timestep", m.opt.timestep, "nu", m.nu)
print("eq types", [int(x) for x in m.eq_type], "eq_active0", [int(x) for x in m.eq_active0] if hasattr(m,'eq_active0') else None)
print("disableflags", m.opt.disableflags, "enableflags", m.opt.enableflags, "solver", m.opt.solver)
r = env.robots[0]; cc = r.composite_controller
for n,c in cc.part_controllers.items():
    print(n, type(c).__name__, "interp", getattr(c,'interpolator',None), getattr(c,'interpolator_pos',None), getattr(c,'interpolator_ori',None), "lite", getattr(c,'lite_physics',None))
print("robot actuators", sum(len(v) for v in r._ref_actuators_indexes_dict.values()), "nu", m.nu)
print("lite_physics env", env.lite_physics, "control_ts", env.control_timestep, "model_ts", env.model_timestep)
for n,o in list(env._observables.items())[:6]:
    print(n, o._sampling_timestep, o._current_delay, o._corrupter.__name__ if hasattr(o._corrupter,'__name__') else o._corrupter, o._delayer)
import collections
print(collections.Counter((o._sampling_timestep, o._current_delay, o._corrupter.__name__, getattr(o._delayer,'__name__',str(o._delayer)), o._filter.__name__) for o in env._observables.values()))
a = action_vec(env, np.zeros(6), -1, np.zeros(3), 0, -1)
t1=time.time()
for i in range(20): env.step(a)
print("20 steps", time.time()-t1)
