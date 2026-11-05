"""Which difference loses the bread-slot contact in replay: one substep of staleness, or Toaster.update_state's
set_lever(1.0) qpos write after the physics? Pass A: plain live (contact as the live checks see it). Pass B: same
actions; inside Toaster.update_state, forward() before the original runs (fresh contact at the post-physics,
pre-write state), and after the step forward() again (post-write state = what replay sees). States compared A vs B."""
import json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from robocasa.models.fixtures.toaster import Toaster  # noqa
from oracle import reset_to  # noqa
from paths import local  # noqa

mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name="GetToastedBread", split="target", render_onscreen=False)
robosuite.make = mk
env.reset()
orig = Toaster.update_state
s = sys.argv[1]; N = 30


def load():
    z = np.load(local(f"{R}/runs/scenes_seen11_v2/GetToastedBread_s{s}.npz"), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"])
    env.unset_ep_meta()


def run(probe):
    load()
    tz = env.toaster; zero = np.zeros(env.action_spec[0].shape)
    for _ in range(5):
        env.step(zero)
    sp = 0 if tz.check_slot_contact(env, "obj", slot_pair=0) else 1
    jn = tz._joint_names[f"lever_{sp}"]; adr = env.sim.model.get_joint_qpos_addr(jn)
    pre = []
    if probe:
        def wrapped(self, e):
            e.sim.forward(); pre.append((bool(self.check_slot_contact(e, "obj", slot_pair=sp)), float(e.sim.data.qpos[adr])))
            return orig(self, e)
        Toaster.update_state = wrapped
    out, st = [], []
    for t in range(N):
        if t < 6:
            tz.set_joint_state(env=env, min=1.0, max=1.0, joint_names=[jn])
        pre.clear() if probe else None
        env.step(zero)
        if probe:
            env.sim.forward()
            out.append(dict(pre_write=pre[-1][0], lever_pre=round(pre[-1][1], 4), post_write=bool(tz.check_slot_contact(env, "obj", slot_pair=sp)),
                            lever_post=round(float(env.sim.data.qpos[adr]), 4)))
        else:
            out.append(dict(stale=bool(tz.check_slot_contact(env, "obj", slot_pair=sp))))
        st.append(env.sim.get_state().flatten().copy())
    Toaster.update_state = orig
    return out, np.array(st)


A, sa = run(False)
B, sb = run(True)
print(json.dumps(dict(scene=s, states_equal=bool(np.array_equal(sa, sb)), max_abs_diff=float(np.abs(sa - sb).max()),
                      stale=[int(x["stale"]) for x in A], pre_write_fresh=[int(x["pre_write"]) for x in B],
                      post_write_fresh=[int(x["post_write"]) for x in B],
                      lever_pre=[x["lever_pre"] for x in B[:8]], lever_post=[x["lever_post"] for x in B[:8]])), flush=True)
