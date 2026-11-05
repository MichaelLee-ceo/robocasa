import sys
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/stages-A")
from common import *
from atomic_steps import TASKS, _burner_under
task = "KettleBoiling"
env = make_env(task)
load(env, task, 13)
st = np.load(f"{R}/runs/20261025_seen480/run/{task}/{task}_s13_single_steps.npz")["states"]
print("n states", len(st))
for f in range(200, len(st)):
    set_state(env, st[f])
    ok = env._check_success()
    on_stove = OU.check_obj_fixture_contact(env, "obj", env.stove)
    loc, d = _burner_under(env)
    ks = env.stove.get_knobs_state(env=env)
    burners_on = [l for l in ks if env.stove.is_burner_on(env=env, burner_loc=l)]
    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    gd = float(np.linalg.norm(eef - env.sim.data.body_xpos[env.obj_body_id[env.objects["obj"].name]]))
    if ok or f % 16 == 0 or f > 300:
        print(f, "succ", ok, "on_stove", on_stove, "nearest", loc, round(d, 3), "on", burners_on, "grip_dist", round(gd, 3),
              "stage0", TASKS[task][0]["check"](env, {}), flush=True)
