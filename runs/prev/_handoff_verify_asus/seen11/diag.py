"""Diagnostics: ScrubCuttingBoard demo contact-timer / sweep / grasp trace, GetToastedBread lever trace.
usage: python -I diag.py scrub EP [EP...] | toast EP [EP...]"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ["ROBOCASA_ROOT"] + "/runs/scripts"))
import robosuite, robocasa  # noqa
import robocasa.utils.object_utils as OU
from robosuite.controllers import load_composite_controller_config
from demo_utils import load_demo
from oracle import reset_to, set_state
from atomic_steps_seen11 import step_segments

mode, eps = sys.argv[1], [int(x) for x in sys.argv[2:]]
task = {"scrub": "ScrubCuttingBoard", "toast": "GetToastedBread"}[mode]
k = robosuite.make(env_name=task, robots="PandaOmron",
                   controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"),
                   has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True,
                   use_object_obs=True, obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))))
res = {}
for ep in eps:
    d = load_demo(task, ep); u = step_segments(d)
    reset_to(k, d["xml"], d["ep_meta"])
    tr = []
    for f in range(len(d["states"])):
        set_state(k, d["states"][f]); k._check_success()
        if mode == "scrub":
            g = bool(OU.check_obj_grasped(k, "sponge"))
            c = bool(k.check_contact(k.objects["sponge"], k.objects["cutting_board"]))
            cc = bool(OU.check_obj_fixture_contact(k, "sponge", k.counter))
            gc = bool(k.check_contact(k.robots[0].gripper["right"], k.objects["sponge"]))
            q = [float(k.sim.data.qpos[k.sim.model.get_joint_qpos_addr(j)]) for j in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")]
            pos = k.board_contact_positions
            sw = float(np.linalg.norm(np.ptp(np.array(pos), 0))) if pos else 0.0
            tr.append((f, g, gc, round(min(q), 4), c, cc, k.board_contact_timer, round(sw, 3), bool(OU.gripper_obj_far(k, "sponge", th=0.15))))
        else:
            st = k.toaster.get_state(k)
            tr.append((f, {str(s): (round(float(v["lever"]), 1), bool(v["turned_on"])) for s, v in st.items()},
                       [k.toaster._num_steps_on[s] for s in k.toaster._slot_pairs]))
    res[ep] = dict(segs=[(s["skill"], s["stage"], s["start"], s["end"]) for s in d["segs"]], units=u)
    if mode == "scrub":
        res[ep]["final"] = tr[-1]
        res[ep]["grasped_frames"] = sum(t[1] for t in tr); res[ep]["gripper_contact_frames"] = sum(t[2] for t in tr)
        res[ep]["board_contact_frames"] = sum(t[4] for t in tr)
        res[ep]["board_contact_while_gripper_contact"] = sum(t[4] and t[2] for t in tr)
        res[ep]["min_finger_q_while_contact"] = min([t[3] for t in tr if t[2]] or [None])
        res[ep]["trace_every20"] = tr[::20]
    else:
        changes = [tr[0]] + [tr[i] for i in range(1, len(tr)) if tr[i][1] != tr[i - 1][1] and (tr[i][1] != tr[i-1][1])]
        res[ep]["lever_changes"] = [c for c in changes][:60]
    print(json.dumps({ep: {kk: v for kk, v in res[ep].items() if kk != "trace_every20"}}, default=str)[:3000], flush=True)
json.dump(res, open(os.path.join(HERE, f"diag_{mode}.json"), "w"), default=str, indent=1)
