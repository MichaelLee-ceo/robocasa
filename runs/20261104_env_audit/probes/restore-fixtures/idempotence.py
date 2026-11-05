"""set_state(s) then get_state() == s ? For every scene of a task, at t = 0, mid and end of the saved baseline episode
(each restored on its own, after a replay up to it so the python state is the episode's). Reports the joints that
update_state() writes. usage: python idempotence.py TASK"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1007/-home-michaellee-mclee-robocasa/13a5c8a4-4ec4-4d74-96b7-fb413333eea4/scratchpad/audit/restore-fixtures")
from common import *  # noqa
task = sys.argv[1]
env = None
res = []
for row in sorted([r for r in rows() if r["task"] == task], key=lambda r: r["tag"])[:int(sys.argv[2])]:
    if env is None:
        env = make_env(task)
    load_scene(env, scene_of(row))
    states = np.load(f"{RUN}/{task}/{row['tag']}_steps.npz")["states"]
    m = env.sim.model
    qadr = {}
    for j in range(m.njnt):
        a = m.jnt_qposadr[j]
        qadr[int(a) + 1] = m.joint_id2name(j)  # flattened state = [time, qpos, qvel, ...]
    bad = []
    marks = {0, len(states) // 2, len(states) - 1}
    for f in range(len(states)):
        set_state(env, states[f]); env._check_success()
        if f in marks:
            got = env.sim.get_state().flatten()
            d = np.nonzero(got != states[f])[0]
            if len(d):
                names = sorted({qadr.get(int(i), f"idx{int(i)}") for i in d})
                bad.append(dict(f=f, n=len(d), names=names[:6], max=float(np.abs(got - states[f]).max())))
    res.append(dict(tag=row["tag"], bad=bad))
nbad = sum(1 for r in res if r["bad"])
print(json.dumps(dict(task=task, scenes=len(res), scenes_with_writes=nbad, examples=[r for r in res if r["bad"]][:3])))
