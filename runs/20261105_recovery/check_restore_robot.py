"""Check loop.restore_robot (the reset to a stage start that moves only the robot back). The future depends on the
integration state (MuJoCo state + the Python graph); derived arrays (contacts, xpos, ...) are recomputed by
robosuite's forward() before every control step, so they are not compared.
1. No-op: restore_robot to a snapshot of the same instant, then the recorded chunks: the trajectory must be
   bit-identical to running them directly from that snapshot.
2. Robot part complete: snapshot at A, replay to B (the robot moves, and here the kettle shifts 3 cm), restore_robot(A),
   then also put every object and fixture joint back as at A. Replaying the chunks after A must then give the same
   trajectory as the continuous replay from A. Any robot state restore_robot missed would show as a difference
   (the time and the observation clock differ: they must not matter for the physics).
usage: python -I check_restore_robot.py [task seed A B M]"""
import gzip, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "envkit"), HERE]
import numpy as np  # noqa: E402
import envkit  # noqa: E402
from loop import ROBOT, restore_robot  # noqa: E402

TASK, SEED, A, B, M = (sys.argv[1], int(sys.argv[2]), *map(int, sys.argv[3:6])) if len(sys.argv) > 5 else \
    ("KettleBoiling", 26, 18, 24, 6)
rec = pickle.load(gzip.open(os.path.join(HERE, "phase0", "records", f"{TASK}_s{SEED}.pkl.gz")))
ep = envkit.Episode(TASK, SEED)
e = ep.kenv
m = e.sim.model
assert all(n.startswith(ROBOT) for n in m.actuator_names), "an actuator that is not the robot's"
wq, wv = [], []
for j in range(m.njnt):
    if not (m.joint_id2name(j) or "").startswith(ROBOT):
        nq, nv = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[int(m.jnt_type[j])]
        wq += range(m.jnt_qposadr[j], m.jnt_qposadr[j] + nq); wv += range(m.jnt_dofadr[j], m.jnt_dofadr[j] + nv)
wq, wv = np.array(wq), np.array(wv)


def traj(chunks):
    out = []
    for c in chunks:
        ep.step(c["actions"])
        out.append(np.concatenate([e.sim.data.qpos, e.sim.data.qvel]).copy())
    return out


def diff(x, y):
    return [f"{float(np.abs(a - b).max()):.1e}" for a, b in zip(x, y)]


for c in rec["chunks"][:A]:
    ep.step(c["actions"])
snapA = ep.snapshot()
ref = traj(rec["chunks"][A:A + M])                       # continuous from A
ep.restore(snapA)
restore_robot(ep, snapA)
print("1. no-op restore_robot, then the same chunks: max |diff| of qpos+qvel per chunk:",
      diff(traj(rec["chunks"][A:A + M]), ref))

ep.restore(snapA)
for c in rec["chunks"][A:B]:
    ep.step(c["actions"])
print(f"   A->B: world joints moved up to {np.abs(e.sim.data.qpos[wq] - snapA.mjdata.qpos[wq]).max():.2e}")
restore_robot(ep, snapA)
d, s = e.sim.data, snapA.mjdata
d.qpos[wq], d.qvel[wv], d.qacc_warmstart[wv] = s.qpos[wq], s.qvel[wv], s.qacc_warmstart[wv]
warm = d.qacc_warmstart.copy()
e.sim.forward()  # it overwrites the warm start (integration state): put it back
d.qacc_warmstart[:] = warm
print("2. restore_robot(A) + world joints of A, then the chunks after A: max |diff| per chunk:",
      diff(traj(rec["chunks"][A:A + M]), ref))
ep.close()
