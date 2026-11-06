"""What the robot did in a recovery trial: replay its record (exact) and print the target's state per chunk end.
DeliverStraw: drawer opening (official measure), hand-to-drawer distance, gripper contact with the drawer.
GetToastedBread: lever value and turned_on per slot pair, the bread in a slot, hand-to-lever distance.
Other tasks: hand-to-target distance and grasp.
usage: python -I inspect_trial.py <record .pkl.gz> [all]   (default: from the failure chunk on; 'all' = every chunk)"""
import gzip, json, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"))
import numpy as np  # noqa: E402
import robocasa.utils.object_utils as OU  # noqa: E402
import envkit, events, stages  # noqa: E402


def body_pos(e, name):
    return np.array(e.sim.data.xpos[e.sim.model.body_name2id(name)])


def row(ep):
    e = ep.kenv
    h = events._eef(e)
    out = dict(t=ep.t, stage=stages.track(ep)["stage"], fingers=round(sum(events._fingers(e)), 3))
    if ep.task == "DeliverStraw":
        jb = e.sim.model.jnt_bodyid[e.sim.model.joint_name2id(f"{e.drawer.name}_slidejoint")]
        out.update(drawer=round(float(e.drawer.get_door_state(env=e)["door"]), 3),
                   hand_drawer=round(float(np.linalg.norm(h - e.sim.data.xpos[jb])), 3),
                   touch_drawer=events._touching(e, "drawer"))
    elif ep.task == "GetToastedBread":
        t = e.toaster
        st = t.get_state(e)
        for sp in t._slot_pairs:
            jn = t._joint_names[f"lever_{sp}"]
            lb = e.sim.model.jnt_bodyid[e.sim.model.joint_name2id(jn)]
            out[f"lever{sp}"] = round(float(st[sp]["lever"]), 2)
            out[f"on{sp}"] = bool(st[sp]["turned_on"])
            out[f"cool{sp}"] = int(t._cooldown[sp])
            out[f"bread_in{sp}"] = bool(t.check_slot_contact(e, "obj", slot_pair=sp))
            out[f"hand_lever{sp}"] = round(float(np.linalg.norm(h - e.sim.data.xpos[lb])), 3)
        out["toaster_on"] = bool(e.toaster_on)
        out["bread_grasped"] = bool(OU.check_obj_grasp(e, "obj") if hasattr(OU, "check_obj_grasp") else False)
    else:
        name = {"KettleBoiling": "obj", "SetUpCuttingStation": "meat"}.get(ep.task, "obj")
        out.update(hand_obj=round(float(np.linalg.norm(h - body_pos(e, e.objects[name].root_body))), 3))
    return out


if __name__ == "__main__":
    r = pickle.load(gzip.open(sys.argv[1]))
    inst = r["instance"]
    start = 0 if "all" in sys.argv[2:] else r["start_chunk"] - 1
    ep = envkit.Episode(inst["task"], inst["seed"])
    for i, c in enumerate(r["chunks"]):
        ep.step(c["actions"])
        if i >= start:
            print(json.dumps(dict(chunk=i, **row(ep))), flush=True)
    ep.close()
