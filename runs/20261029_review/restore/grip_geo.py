"""Contact-side test for the gripper ramp (plain mujoco on the scene MJCF): at a boundary where the fingers touch
something, is the contact pushing each finger toward closing (the obstacle is outside the fingers: an open gripper
squeezed, e.g. pressing on the microwave door) or toward opening (the obstacle is between the fingers: a grasp)?
score = sum over finger contacts of sign(f . closing direction of that finger), f = direction of the contact normal
force on the finger. > 0: squeezed (evidence for an open command); < 0: grasp (evidence for a close command).
Computed for the out/grip_table.npz rows with a finger contact and a gap of 5-78.5 mm (all non-held ones, every 4th
held one). Writes out/grip_geo.json {key: score, n_in, n_out}.
usage: python -I grip_geo.py"""
import json, os, sys
from collections import defaultdict
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_boundaries import Model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"


def finger_side(M, s):
    m, d = M.m, M.d
    d.qpos[:] = s[1:1 + m.nq]; d.qvel[:] = s[1 + m.nq:1 + m.nq + m.nv]
    mujoco.mj_forward(m, d)
    j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_right_finger_joint1")
    j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_right_finger_joint2")
    close1 = -np.array(d.xaxis[j1])  # joint1 opens with +q
    close2 = np.array(d.xaxis[j2])   # joint2 opens with -q
    n_in = n_out = 0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        n = np.array(c.frame[:3])
        for fb, ob, fdir in ((b1, b2, -n), (b2, b1, n)):
            name = M.bname[fb]
            other = M.bname[M.root[ob]] or ""
            if not name.startswith("gripper0_") or other.startswith(("robot0", "gripper0", "mobilebase0")):
                continue
            if "leftfinger" in name or "joint1_tip" in name:
                cl = close1
            elif "rightfinger" in name or "joint2_tip" in name:
                cl = close2
            else:
                continue
            if float(np.dot(fdir, cl)) > 0:
                n_out += 1   # pushes the finger closed: obstacle outside
            else:
                n_in += 1    # pushes the finger open: obstacle between the fingers
    return n_in, n_out


def main():
    Z = np.load(f"{HERE}/out/grip_table.npz", allow_pickle=True)
    keys, held, obj, fix, hist = Z["key"], Z["held"], Z["obj"], Z["fix"], Z["hist"]
    g0 = hist[:, -1]
    sel = ((obj | fix) & (g0 >= 0.005) & (g0 < 0.0785) & ~held) | (held & (np.arange(len(keys)) % 4 == 0))
    idx = np.nonzero(sel)[0]
    scenes = {}
    for p in sorted(__import__("glob").glob(f"{D}/jobs*.json")):
        for j in json.load(open(p)):
            scenes[j["tag"]] = j["scene"]
    by_tag = defaultdict(list)
    for i in idx:
        by_tag[keys[i].split("|")[1]].append(i)
    out = {}
    for n_tag, (tag, ii) in enumerate(sorted(by_tag.items())):
        M = Model(scenes[tag])
        cache = {}
        for i in ii:
            d, _, t, c, b = keys[i].split("|")
            f = f"{D}/{d}/roll/{tag}_t{t}_c{c}_m0.npz"
            if f not in cache:
                cache = {f: np.load(f)["states"]}
            n_in, n_out = finger_side(M, cache[f][int(b)])
            out[keys[i]] = (n_in, n_out)
        print(n_tag, tag, len(ii), flush=True)
    json.dump(out, open(f"{HERE}/out/grip_geo.json", "w"))


if __name__ == "__main__":
    main()
