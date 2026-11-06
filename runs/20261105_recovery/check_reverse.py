"""Does reversing executed chunks bring the robot back? Replays a loop record (exact) and, at every 'back' intervention
of k chunks at chunk n, compares the robot after the reversal (after n + k chunks) with the robot before the reversed
chunks (after n - k chunks): hand position (world), arm joints, base pose, fingers; and how far the robot had moved in
those k chunks (after n - k vs after n).
usage: python -I check_reverse.py <loop record .pkl.gz> ..."""
import gzip, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import numpy as np  # noqa: E402
import envkit, events  # noqa: E402


def robot(e):
    m, d = e.sim.model, e.sim.data
    names = [m.joint_id2name(j) for j in range(m.njnt)]
    arm = [d.qpos[m.get_joint_qpos_addr(n)] for n in names if n and n.startswith("robot0_joint")]
    base = [d.qpos[m.get_joint_qpos_addr(n)] for n in names if n and n.startswith("mobilebase0_joint")]
    return dict(hand=events._eef(e), arm=np.array(arm), base=np.array(base), fingers=np.array(events._fingers(e)))


def diff(a, b):
    return dict(hand_cm=round(100 * float(np.linalg.norm(a["hand"] - b["hand"])), 1),
                arm_rad=round(float(np.abs(a["arm"] - b["arm"]).max()), 3),
                base=np.round(a["base"] - b["base"], 3).tolist(),
                fingers=round(float(np.abs(a["fingers"] - b["fingers"]).max()), 4))


for path in sys.argv[1:]:
    r = pickle.load(gzip.open(path))
    out = r["out"]
    backs = [x for x in out["log"] if x["event"] == "intervene" and x.get("reversed")]
    need = {i for x in backs for i in (x["chunk"] - x["reversed"], x["chunk"], x["chunk"] + x["reversed"])}
    ep = envkit.Episode(out["task"], out["seed"])
    st = {0: robot(ep.kenv)} if 0 in need else {}
    for i, c in enumerate(r["chunks"]):
        if i >= max(need):
            break
        ep.step(c["actions"])
        if i + 1 in need:
            st[i + 1] = robot(ep.kenv)
    ep.close()
    print(os.path.basename(path))
    for x in backs:
        n, k = x["chunk"], x["reversed"]
        if n + k not in st:  # the episode ended during the reversal
            continue
        print(f"  back {k} at {n}: moved {diff(st[n], st[n - k])}  ->  after reversal vs before: {diff(st[n + k], st[n - k])}")
