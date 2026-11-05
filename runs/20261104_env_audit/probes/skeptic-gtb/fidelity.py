"""Live vs replayed slot-floor contact while the bread's pair is on (CPU). Scene: bread seated, lever set to 1.0, zero
actions. Per step: live contact (data.contact left by env.step, which reward()->_check_success saw) and the official
latch env.toaster_on; then replay the saved states with oracle.set_state and recompute contact."""
import json, sys
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts"); sys.path.insert(0, f"{R}/runs/20261025_seen480")
from analyze_stages import replay_env
import atomic_eval  # noqa
from atomic_eval import load_scene
from oracle import set_state

scene = sys.argv[1]
env = replay_env("GetToastedBread")
path = f"{R}/runs/scenes_seen11_v2/{scene}.npz"
load_scene(env, path); env.toaster_on = False
a0 = np.zeros(env.action_dim)
npair = len(env.toaster.get_state(env))
for _ in range(16):
    env.step(a0)
c = [env.toaster.check_slot_contact(env, "obj", slot_pair=i) for i in range(npair)]
bs = int(np.argmax(c)); print("pairs", npair, "contact", c, "bread pair", bs)
states = [env.sim.get_state().flatten().copy()]
env.toaster.set_lever(env, bs, 1.0)
live, on, bz = [], [], []
for t in range(560):
    env.step(a0)
    states.append(env.sim.get_state().flatten().copy())
    live.append(bool(env.toaster.check_slot_contact(env, "obj", slot_pair=bs)))
    on.append(bool(env.toaster.get_state(env, slot_pair=bs)["turned_on"]))
    bz.append(float(env.sim.data.body_xpos[env.obj_body_id["obj"]][2]))
print("live: toaster_on latch", env.toaster_on, "contact frames while on", sum(l and o for l, o in zip(live, on)), "/", sum(on),
      "first on", on.index(True) if True in on else None, "bread dz", round(bz[-1] - bz[0], 4), "min z", round(min(bz) - bz[0], 4))
# replay
z = np.load(path, allow_pickle=True)
from oracle import reset_to
reset_to(env, str(z["xml"]), json.loads(str(z["ep_meta"])))
env.toaster_on = False
rep = []
for f, s in enumerate(states):
    set_state(env, s)
    if f > 0:
        rep.append(bool(env.toaster.check_slot_contact(env, "obj", slot_pair=bs)))
agree = sum(a == b for a, b in zip(live, rep))
print("replay: contact frames", sum(rep), "live contact frames", sum(live), "agree", agree, "/", len(live))
runs = lambda m: [i for i in range(1, len(m)) if m[i] != m[i - 1]]
print("live contact transitions", runs(live)[:20]); print("replay contact transitions", runs(rep)[:20])
