"""Joint layout of every scene used by a job (no simulation: the scene XML is only compiled with mujoco).
The flattened sim state is [time, qpos (nq), qvel (nv)] (robosuite MjSimState.flatten); this records, per scene, where
the robot's joints and the free-joint objects sit in it, so the state arrays can be read without building the env.
Writes layout.json next to this file. usage: python -I layout.py [workers=3]"""
import glob, json, os, sys
from multiprocessing import get_context
import numpy as np

R = "/home/michaellee/mclee/robocasa"
DATA = f"{R}/runs/20261026_critic_data"
HERE = os.path.dirname(os.path.abspath(__file__))


def one(scene):
    import mujoco
    z = np.load(scene, allow_pickle=True)
    xml = str(z["xml"]).replace("/home/michaellee/Desktop/robocasa", R)
    m = mujoco.MjModel.from_xml_string(xml)
    fp = json.loads(str(z["fp"]))
    joints = {}
    for i in range(m.njnt):
        joints[m.joint(i).name] = dict(type=int(m.jnt_type[i]), qpos=int(m.jnt_qposadr[i]), dof=int(m.jnt_dofadr[i]),
                                      range=[float(x) for x in m.jnt_range[i]], axis=[float(x) for x in m.jnt_axis[i]],
                                      body=m.body(m.jnt_bodyid[i]).name)
    return scene, dict(nq=int(m.nq), nv=int(m.nv), na=int(m.na), nu=int(m.nu), nmocap=int(m.nmocap),
                       fp_nq=int(fp["nq"]), objects=list(fp["objects"]), joints=joints,
                       state_len=1 + int(m.nq) + int(m.nv))


def main():
    scenes = sorted({j["scene"] for f in glob.glob(f"{DATA}/jobs*.json") for j in json.load(open(f))})
    out = {}
    with get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for s, info in pool.imap_unordered(one, scenes):
            out[s] = info
            print(os.path.basename(s), info["nq"], info["nv"], info["na"], flush=True)
    json.dump(out, open(os.path.join(HERE, "layout.json"), "w"))


if __name__ == "__main__":
    main()
