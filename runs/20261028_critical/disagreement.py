"""Source A at the decision points already collected: do the 8 candidate chunks disagree more where the chunk matters?

From each candidate's first chunk (roll/<tag>_t<t>_c<i>_m0.npz, chunk 0; 16 steps), per decision point:
  grip    mean over the 16 steps of the variance across candidates of the gripper_close command (0 = all agree)
  pos     spread of where the candidates take the end effector: RMS distance of each candidate's summed
          end_effector_position command from the mean of the 8 (action units)
  rot     the same for end_effector_rotation
  base    the same for base_motion
  mode    mean over the steps of the variance across candidates of control_mode (arm vs base)
  pos_rel pos divided by the candidates' mean summed end_effector_position length: disagreement in direction
  grip_end variance across candidates of the gripper_close command at the chunk's last step (open vs closed after it)
The policy computes all of these when it samples, so a trigger built on them is deployable.
The chunks are read from the rollouts of the coarse row in use (data.py), the same collection as the labels.
Writes disagreement.jsonl and prints the chunk effect (as effect_by_event.py) by quartile of each measure, all tasks
pooled and per task. usage: python -I disagreement.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
from data import OUT, TASKS  # noqa: E402
from effect_by_event import points, summary  # noqa: E402

MEASURES = ["grip", "pos", "rot", "base", "mode", "pos_rel", "grip_end"]


def roll_dir(task, tag, t):
    """the rollouts of the coarse row in use at (tag, t); points() carries the same as p["roll"]"""
    return data.roll(data.rows(task)[(tag, t)])


def spread(x):
    return float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()))


def measures(chunks):
    g = np.stack([c["action.gripper_close"][0, :, 0] for c in chunks])
    m = np.stack([c["action.control_mode"][0, :, 0] for c in chunks])
    P = np.stack([c["action.end_effector_position"][0].sum(0) for c in chunks])
    return dict(grip=float(g.var(0).mean()), mode=float(m.var(0).mean()), pos=spread(P),
                pos_rel=spread(P) / (float(np.linalg.norm(P, axis=1).mean()) + 1e-6), grip_end=float(g[:, -1].var()),
                rot=spread(np.stack([c["action.end_effector_rotation"][0].sum(0) for c in chunks])),
                base=spread(np.stack([c["action.base_motion"][0].sum(0) for c in chunks])))


def main():
    pts, rows = points(), []
    for p in pts:
        chunks = [np.load(f"{p['roll']}/{p['tag']}_t{p['t']}_c{i}_m0.npz") for i in range(len(p["y"]))]
        p.update(measures(chunks))
        rows.append({k: p[k] for k in ["task", "tag", "t", "k", "cb", "pd"] + MEASURES})
    with open(os.path.join(OUT, "disagreement.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{len(pts)} decision points")
    for m in MEASURES:
        print(f"--- {m}: bins over all tasks, then the lowest and highest bin per task")
        for lab, scope in [("all tasks", None)] + [(t, t) for t in TASKS]:
            g = [p for p in pts if scope is None or p["task"] == scope]
            v = np.array([p[m] for p in g])
            if (v == 0).mean() > 0.25:  # most candidates agree exactly: zero, then halves of the rest
                edges = [-np.inf, 0, np.median(v[v > 0]) if (v > 0).any() else 0, np.inf]
            else:
                edges = [-np.inf, *np.quantile(v, [0.25, 0.5, 0.75]), np.inf]
            nb = len(edges) - 1
            for i in range(nb):
                sel = [p for p in g if edges[i] < p[m] <= edges[i + 1]]
                if scope is None or i in (0, nb - 1):
                    print(f"  {lab:21s} bin {i + 1}/{nb} ({edges[i]:.3g}, {edges[i + 1]:.3g}]".ljust(60), summary(sel))


if __name__ == "__main__":
    main()
