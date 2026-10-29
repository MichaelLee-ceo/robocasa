"""Feature scan of the chunk boundaries inside saved rollouts (candidate restore points for the continuous-vs-restore
matrix). Plain mujoco on the scene MJCF (no env, no rendering): set qpos/qvel, mj_forward, read contacts.
Per boundary b = 16c: control mode of chunk c (arm / base / switch inside), mode at the end of chunk c-1, base yaw since
the episode start (mobilebase0_joint_mobile_yaw), base speed, finger gap, true gripper ramp (from the actions) and
settle's v4 inference, what the fingers touch (objects held with both fingers, fixtures), fixture joints moving in
chunk c. usage: python -I scan_boundaries.py <task> -> out/boundaries_<task>.jsonl"""
import glob, json, os, sys
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_scan import DIRS, rows, infer, rule_of, gap  # noqa: E402

R = "/home/michaellee/mclee/robocasa/runs"
D = f"{R}/20261026_critic_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


class Model:
    def __init__(self, scene):
        z = np.load(scene, allow_pickle=True)
        xml = str(z["xml"]).replace("/home/michaellee/Desktop/robocasa/", "/home/michaellee/mclee/robocasa/")
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        self.root = m.body_rootid
        self.free = {}
        self.fx = []
        for j in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if m.jnt_type[j] == 0:
                self.free[n] = int(m.jnt_qposadr[j])
            elif not n.startswith(("robot0", "mobilebase0", "gripper0")):
                self.fx.append((n, int(m.jnt_qposadr[j])))
        self.left = [i for i, n in enumerate(self.bname) if n.startswith("gripper0_") and ("leftfinger" in n or "finger1" in n or "left" in n)]
        self.right = [i for i, n in enumerate(self.bname) if n.startswith("gripper0_") and ("rightfinger" in n or "finger2" in n or "right_inner" in n)]
        self.grip = [i for i, n in enumerate(self.bname) if n.startswith("gripper0_")]

    def contacts(self, s):
        m, d = self.m, self.d
        d.qpos[:] = s[1:1 + m.nq]; d.qvel[:] = s[1 + m.nq:1 + m.nq + m.nv]
        mujoco.mj_forward(m, d)
        touch = {}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for x, y in ((b1, b2), (b2, b1)):
                if x in self.grip and y not in self.grip:
                    other = self.bname[self.root[y]] or self.bname[y]
                    if other.startswith(("robot0", "mobilebase0")):
                        continue
                    touch.setdefault(other, set()).add(self.bname[x])
        return {k: sorted(v) for k, v in touch.items()}


def main():
    for task in sys.argv[1].split(","):
        scan(task, int(sys.argv[2]) if len(sys.argv) > 2 else 1)


def scan(task, every):
    out = open(f"{OUT}/boundaries_{task}.jsonl", "w")
    models = {}
    scenes = {}
    for p in sorted(glob.glob(f"{D}/jobs*.json")):
        for j in json.load(open(p)):
            scenes[j["tag"]] = j["scene"]
    todo = sorted(((d, r) for d in DIRS[task] for r in rows(d)), key=lambda x: (x[1]["tag"], x[0], x[1]["t"]))
    for d, r in todo:
        if True:
            tag, t = r["tag"], r["t"]
            scene = scenes[tag]
            if scene not in models:
                models = {scene: Model(scene)}
            M = models[scene]
            ep = np.load(f"{R}/20261025_seen480/run/{task}/{tag}_steps.npz")["states"]
            c0 = infer(ep[t], ep[t - 1] if t else None, rule_of(r.get("hidden")))
            for ci in range(0, len(r["cands"]), every):
                f = f"{D}/{d}/roll/{tag}_t{t}_c{ci}_m0.npz"
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                S = z["states"]
                A = z["action.gripper_close"][:, :, 0].reshape(-1)
                mode = z["action.control_mode"][:, :, 0]
                bm = z["action.base_motion"]
                cmd = np.where(A < 0.5, -1.0, 1.0)
                c = [c0]
                for a in cmd[: len(S) - 1]:
                    c.append(float(np.clip(c[-1] + 0.2 * a, -1, 1)))
                if S.shape[1] != 1 + M.m.nq + M.m.nv:
                    print("state length mismatch", tag, S.shape, M.m.nq, M.m.nv); continue
                for b in range(16, len(S) - 16, 16):
                    ch = b // 16
                    mo = mode[ch] >= 0.5
                    tc = M.contacts(S[b])
                    held = [o for o, fs in tc.items() if o.endswith("_main") and o[:-5] + "_joint0" in M.free
                            and any(("leftfinger" in x or "joint1_tip" in x) for x in fs)
                            and any(("rightfinger" in x or "joint2_tip" in x) for x in fs)]
                    fxm = [n for n, q in M.fx if abs(S[b + 16][1 + q] - S[b][1 + q]) > 0.01]
                    rec = dict(task=task, dir=d, tag=tag, t=t, k=r["k"], c=ci, m=0, b=b, n=len(S) - 1,
                               mode=("base" if mo.all() else "arm" if not mo.any() else "switch"),
                               prev_mode_end=int(mode[ch - 1, -1] >= 0.5),
                               yaw_deg=round(float(np.degrees(S[b][3])), 1),
                               base_speed=round(float(np.linalg.norm(S[b][1 + M.m.nq:1 + M.m.nq + 3])), 3),
                               base_cmd=round(float(np.abs(bm[ch, :, :3]).max()), 3),
                               gap_mm=round(gap(S[b]) * 1000, 2), dgap_mm=round((gap(S[b]) - gap(S[b - 1])) * 1000, 3),
                               ramp_true=round(c[b], 2), ramp_v4=infer(S[b], S[b - 1], "v4"),
                               ramp_v3=infer(S[b], S[b - 1], "v3"), cmd_next=float(cmd[b]),
                               touch=tc, held=held, fixture_moving=fxm)
                    out.write(json.dumps(rec) + "\n")
            out.flush()
    print(task, "done", flush=True)


if __name__ == "__main__":
    main()
