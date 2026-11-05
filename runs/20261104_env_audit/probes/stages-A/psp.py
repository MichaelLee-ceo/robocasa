import sys
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/stages-A")
from common import *
from atomic_steps import TASKS
task = "PreSoakPan"
env = make_env(task)
for s in (16, 19):
    load(env, task, s)
    st = np.load(f"{R}/runs/20261025_seen480/run/{task}/{task}_s{s}_single_steps.npz")["states"]
    for f in (0, 16, 100, 300):
        set_state(env, st[f])
        ins = {o: bool(OU.obj_inside_of(env, o, env.sink, partial_check=False)) for o in ("obj1", "obj2")}
        far = {o: bool(OU.gripper_obj_far(env, o)) for o in ("obj1", "obj2")}
        pos = {o: np.round(env.sim.data.body_xpos[env.obj_body_id[o]], 3).tolist() for o in ("obj1", "obj2")}
        print(s, f, "inside", ins, "far", far, "pos", pos, "lang", {o: OU.get_obj_lang(env, o) for o in ("obj1","obj2")},
              "water", env.sink.get_handle_state(env=env)["water_on"], flush=True)
