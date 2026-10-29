"""Joint-address helpers for the flattened sim states (time, qpos, qvel, ...), built from a scene's MJCF with plain
mujoco (no env, no rendering). Cached per scene in model_index.json."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "model_index.json")


def index(scene):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if scene in cache:
        return cache[scene]
    import mujoco
    z = np.load(scene, allow_pickle=True)
    xml = str(z["xml"]).replace("/home/michaellee/Desktop/robocasa/", "/home/michaellee/mclee/robocasa/")
    m = mujoco.MjModel.from_xml_string(xml)
    J = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        J[n] = dict(type=int(m.jnt_type[j]), qpos=int(m.jnt_qposadr[j]), qvel=int(m.jnt_dofadr[j]),
                    range=[float(x) for x in m.jnt_range[j]])
    out = dict(nq=int(m.nq), nv=int(m.nv), na=int(m.na), nu=int(m.nu), nmocap=int(m.nmocap), neq=int(m.neq),
               joints=J, state_len=int(z["state"].shape[0]))
    cache[scene] = out
    json.dump(cache, open(CACHE, "w"))
    return out


if __name__ == "__main__":
    o = index(sys.argv[1])
    print({k: v for k, v in o.items() if k != "joints"})
    for n, j in o["joints"].items():
        if n.startswith(("robot0", "mobilebase0", "gripper0")):
            print(n, j)
