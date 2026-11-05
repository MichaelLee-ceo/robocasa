"""GetToastedBread: same trajectory, live vs replay. Live: env.step (reward()->_check_success, update_state), then a
second _check_success (gym wrapper). Record per step: turned_on, slot contact as the live checks see it (stale
kinematics after the last substep), toaster_on, lever qpos, and the saved state (as atomic_eval saves it, after step).
Replay: reset_to the same scene, then for each saved state set_state + _check_success (stage_trace / validate)."""
import json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robosuite  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
from paths import local  # noqa

mk = robosuite.make
robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
env = create_env(env_name="GetToastedBread", split="target", render_onscreen=False)
robosuite.make = mk
env.reset()


def load(path):
    z = np.load(local(path), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"])
    env.unset_ep_meta()


N = int(sys.argv[1])
for s in sys.argv[2:]:
    path = f"{R}/runs/scenes_seen11_v2/GetToastedBread_s{s}.npz"
    load(path)
    tz = env.toaster
    pairs = list(range(len(tz.get_state(env))))
    init = dict(toaster_on=bool(env.toaster_on), on=[bool(tz.get_state(env, slot_pair=i)["turned_on"]) for i in pairs])
    zero = np.zeros(env.action_spec[0].shape)
    for _ in range(5):
        env.step(zero)
    c5 = [i for i in pairs if tz.check_slot_contact(env, "obj", slot_pair=i)]
    sp = c5[0] if c5 else 0
    jn = tz._joint_names[f"lever_{sp}"]
    lever_adr = env.sim.model.get_joint_qpos_addr(jn)
    states, live = [], []
    for t in range(N):
        if t < 6:
            tz.set_joint_state(env=env, min=1.0, max=1.0, joint_names=[jn])
        _, rew, _, _ = env.step(zero)
        c_stale = bool(tz.check_slot_contact(env, "obj", slot_pair=sp))
        ok2 = bool(env._check_success())
        live.append(dict(rew=float(rew), on=bool(tz.get_state(env, slot_pair=sp)["turned_on"]), c=c_stale,
                         latch=bool(env.toaster_on), lever=float(env.sim.data.qpos[lever_adr])))
        states.append(env.sim.get_state().flatten().copy())
    load(path)
    rep = []
    for st in states:
        set_state(env, st)
        env._check_success()
        rep.append(dict(on=bool(tz.get_state(env, slot_pair=sp)["turned_on"]),
                        c=bool(tz.check_slot_contact(env, "obj", slot_pair=sp)), latch=bool(env.toaster_on)))
    f = lambda L, k: next((t for t, x in enumerate(L) if x[k]), None)
    print(json.dumps(dict(scene=s, init=init, slot=sp, contact_after5=c5,
                          live=dict(first_on=f(live, "on"), n_on_contact=sum(x["on"] and x["c"] for x in live),
                                    n_on=sum(x["on"] for x in live), latched_at=f(live, "latch")),
                          replay=dict(first_on=f(rep, "on"), n_on_contact=sum(x["on"] and x["c"] for x in rep),
                                      n_on=sum(x["on"] for x in rep), latched_at=f(rep, "latch")),
                          lever_live_first10=[round(x["lever"], 4) for x in live[:10]],
                          contact_live_first12=[int(x["c"]) for x in live[:12]],
                          contact_replay_first12=[int(x["c"]) for x in rep[:12]])), flush=True)
