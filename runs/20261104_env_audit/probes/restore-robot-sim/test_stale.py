"""After env.step() (lite_physics: step1 -> controller -> step2 per substep) mjData's derived quantities (xpos, contacts,
...) are those of the state BEFORE the last substep's integration, while qpos/qvel are after it. The live reward /
update_state / stage checks see the former; set_state + forward (replay, restore, stage_trace, labeler checks) see the
latter. Measure the gap on a live fake-policy run."""
import sys, json
sys.path.insert(0, sys.argv[1])
from harness import *
from fake import wrap, FakeClient, seedf
from atomic_eval import load_scene
from atomic_steps import TASKS
task = sys.argv[2]
env = make_env(task); base = wrap(env)
load_scene(env, scene_path(task, 0))
steps = TASKS[task]
cl = FakeClient(); sf = seedf("stale")
r = env.robots[0]; eef = r.eef_site_id["right"]
def contacts(e):
    d = e.sim.data
    return {(min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2))) for c in d.contact[:d.ncon]}
def checks(e):
    return [bool(s["check"](e, {})) for s in steps] + [bool(e._check_success())]
dpos, ncdiff, nchk, n = [], 0, 0, 0
obs = None
for ci in range(8):
    req = {"_seed": sf(ci), "state.x": np.zeros(1)}
    a = cl.get_action(req)
    for i in range(16):
        act = {k: np.asarray(v)[0][i] for k, v in a.items() if k.startswith("action")}
        base.step(act)
        stale_p = np.array(env.sim.data.site_xpos[eef]); cs = contacts(env); ck = checks(env)
        ws = np.array(env.sim.data.qacc_warmstart)
        x = env.sim.get_state().flatten()
        env.sim.forward()
        fresh_p = np.array(env.sim.data.site_xpos[eef]); cf = contacts(env); ckf = checks(env)
        env.sim.data.qacc_warmstart[:] = ws
        assert np.array_equal(env.sim.get_state().flatten(), x)
        dpos.append(1000 * np.linalg.norm(stale_p - fresh_p)); ncdiff += cs != cf; nchk += ck != ckf; n += 1
print("RESULT", json.dumps(dict(task=task, steps=n, eef_stale_vs_fresh_mm_mean=round(float(np.mean(dpos)), 3),
                                eef_mm_max=round(float(np.max(dpos)), 3), steps_contact_set_differs=int(ncdiff), steps_check_differs=int(nchk))))
