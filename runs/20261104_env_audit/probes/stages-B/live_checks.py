"""(1) env-level task state (StirVegetables.success_time / prev_obj_pos_*, GetToastedBread.toaster_on) through
hidden_state.capture + atomic_eval.restore_node; (2) GetToastedBread: does the official toaster_on latch fire in live
stepping when the lever is pushed down with the bread in its slot (frozen scene)?"""
import json, os, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
sys.path.insert(0, f"{R}/third_party/Isaac-GR00T")
import robosuite  # noqa
from robocasa.utils.env_utils import create_env  # noqa
from oracle import reset_to, set_state  # noqa
import hidden_state  # noqa
from paths import local  # noqa


def make(task):
    mk = robosuite.make
    robosuite.make = lambda **kw: mk(**dict(kw, has_offscreen_renderer=False, use_camera_obs=False))
    try:
        env = create_env(env_name=task, split="target", render_onscreen=False)
    finally:
        robosuite.make = mk
    env.reset()
    return env


def load(env, path):
    z = np.load(local(path), allow_pickle=True)
    reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])), z["state"])
    env.unset_ep_meta()
    return z["state"].copy()


def restore_node(kenv, start_state, hidden=None):  # verbatim copy of atomic_eval.restore_node (avoids its gym imports)
    set_state(kenv, start_state)
    if hidden is not None and "obs_value" in hidden:
        kenv.sim.set_state_from_flattened(start_state)
        kenv.sim.forward()
    if hidden is not None:
        hidden_state.restore(kenv, hidden)
        if "obs_value" in hidden and not np.array_equal(kenv.sim.get_state().flatten(), start_state):
            raise RuntimeError("restore_node: the sim state differs from the node after the restore")


which = sys.argv[1]
if which == "stir":
    env = make("StirVegetables")
    s0 = load(env, f"{R}/runs/scenes_seen11_v2/StirVegetables_s0.npz")
    zero = np.zeros(env.action_spec[0].shape)
    for _ in range(3):
        env.step(zero)
    node = env.sim.get_state().flatten().copy()
    h = hidden_state.capture(env, obs=True)
    print("py_state keys:", sorted(hidden_state.py_state(env)), " 'success_time' captured:",
          any("success_time" in str(v) for v in h.get("fixtures", {}).values()))
    print("at node: success_time", env.success_time, "_check_success", env._check_success())
    env.success_time = 7  # what an earlier branch left behind
    env.prev_obj_pos_veg1 = np.array([9.0, 9.0, 9.0])
    restore_node(env, node, h)
    print("after restore_node to the node: success_time", env.success_time, "_check_success", env._check_success(),
          "prev_obj_pos_veg1", getattr(env, "prev_obj_pos_veg1", None))
elif which == "toast_restore":
    env = make("GetToastedBread")
    s0 = load(env, f"{R}/runs/scenes_seen11_v2/GetToastedBread_s0.npz")
    h = hidden_state.capture(env, obs=True)
    print("at t=0: toaster_on", env.toaster_on)
    env.toaster_on = True  # an earlier branch toasted
    restore_node(env, s0, h)
    print("after restore_node to t=0: toaster_on", env.toaster_on, "(the official latch survives the restore)")
elif which == "toast_live":
    for s in sys.argv[2:]:
        env = make("GetToastedBread")
        load(env, f"{R}/runs/scenes_seen11_v2/GetToastedBread_s{s}.npz")
        tz = env.toaster
        st = tz.get_state(env)
        cont = [i for i in range(len(st)) if tz.check_slot_contact(env, "obj", slot_pair=i)]
        zero = np.zeros(env.action_spec[0].shape)
        for _ in range(5):
            env.step(zero)
        cont5 = [i for i in range(len(st)) if tz.check_slot_contact(env, "obj", slot_pair=i)]
        sp = cont5[0] if cont5 else (cont[0] if cont else 0)
        jn = tz._joint_names[f"lever_{sp}"]
        log = []
        for t in range(40):
            if t < 6:  # push the lever down like a finger would: write the joint to its 'on' end before the step
                tz.set_joint_state(env=env, min=1.0, max=1.0, joint_names=[jn])
            env.step(zero)
            stt = tz.get_state(env, slot_pair=sp)
            log.append((t, bool(stt["turned_on"]), bool(tz.check_slot_contact(env, "obj", slot_pair=sp)), bool(env.toaster_on)))
        print(json.dumps(dict(scene=s, slot_pairs=len(st), contact_t0=cont, contact_after5=cont5, pushed=sp,
                              first_on=next((t for t, on, c, l in log if on), None),
                              n_on_with_contact=sum(on and c for t, on, c, l in log),
                              latched_at=next((t for t, on, c, l in log if l), None))), flush=True)
