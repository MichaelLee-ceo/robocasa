"""GetToastedBread: why does the official success fail in replay for some demos? (read-only diagnostic)
usage: python -I toast_diag.py <ep> [<ep> ...]"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
R = "/home/michaellee/mclee/robocasa"
sys.path.insert(0, f"{R}/runs/scripts")
import robocasa.scripts.dataset_scripts.add_subtask_labels_to_lerobot as L  # noqa
import robocasa.utils.object_utils as OU  # noqa
from demo_utils import load_demo  # noqa
from oracle import reset_to, set_state  # noqa
from atomic_steps import TASKS  # noqa

task = "GetToastedBread"
lerobot = Path(sorted(glob.glob(f"{R}/datasets/v1.0/target/composite/{task}/*/lerobot"))[-1])
env = L.build_env(json.loads((lerobot / "extras" / "dataset_meta.json").read_text())["env_args"])
for ep in map(int, sys.argv[1:]):
    d = load_demo(task, ep)
    reset_to(env, d["xml"], d["ep_meta"])
    S = {}
    on_frames, contact_on, latch_f = [], [], None
    first_toast_done = None
    for f in range(len(d["states"])):
        set_state(env, d["states"][f])
        # contact BEFORE update_state's effects is not reproducible; record what replay sees
        st = env.toaster.get_state(env)
        ons = [sp for sp, v in st.items() if v["turned_on"]]
        cont = [i for i in range(len(st)) if env.toaster.check_slot_contact(env, "obj", slot_pair=i)]
        if ons:
            on_frames.append(f)
            if set(ons) & set(cont):
                contact_on.append(f)
        env._check_success()
        if latch_f is None and env.toaster_on:
            latch_f = f
        if first_toast_done is None and TASKS[task][0]["check"](env, S):
            first_toast_done = f
    in_plate = bool(OU.check_obj_in_receptacle(env, "obj", "plate"))
    far = bool(OU.gripper_obj_far(env, "obj"))
    print(json.dumps(dict(ep=ep, length=len(d["states"]), on_first=on_frames[0] if on_frames else None,
                          on_last=on_frames[-1] if on_frames else None, n_on=len(on_frames),
                          n_contact_while_on=len(contact_on), env_toaster_on_latched_at=latch_f,
                          our_stage0_at=first_toast_done, S=S, end_in_plate=in_plate, end_gripper_far=far,
                          end_success=bool(env._check_success()))), flush=True)
