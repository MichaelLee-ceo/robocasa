"""Bug 4 residue: which current points (rows collected before the settled-v4 fix) had the gripper ramp restored on the
wrong side? settle infers the commanded side from the finger gap g at t (and g at t-1):
  old rule (rows 'settled', 'settled-basefix', 'settled-v3'): opening if g >= 1.9 * init gap - 4 mm (= 75.2 mm) or widening
  new rule ('settled-v4'): opening if g >= joint-limit gap - 1.5 mm (= 78.5 mm) or widening
(widening = g - g(t-1) > 0.5 mm). They disagree only for 75.2 <= g < 78.5 mm without widening: the old rule restored an
opening ramp while the fingers were closed on (or closing towards) something. The gap is the trace's grip_width
(|q1| + |q2|, the same quantity, 0.1 mm resolution). For every such point: V, its neighbours, whether the fingers touch
an object at t, and the finger gap of the rollouts after 4 and 8 steps vs the main run's.
Writes grip_rule.json. usage: python grip_rule.py"""
import gzip, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, SEEN, TASKS3, coarse, dense, rolls, roll_path  # noqa: E402

INIT_GAP = 2 * 0.020833  # PandaGripper init_qpos [0.020833, -0.020833]
OLD_OPEN = 1.9 * INIT_GAP - 0.004
NEW_OPEN = 0.080 - 0.0015
FINGER = (12, 13)  # the two finger joints in the flattened state [time, qpos, qvel]: qpos 11, 12 in all three tasks (checked
# against the trace's grip_width on 3 episodes, max error 5e-5 = its rounding)


def main():
    out = []
    traces, mains = {}, {}
    for task in TASKS3:
        pts = {**{k: ("coarse", r) for k, r in coarse(task).items()}, **{k: ("dense", r) for k, r in dense(task).items()}}
        allV = {}
        for (tag, t), (src, r) in pts.items():
            allV.setdefault((tag, r["k"]), []).append((t, np.mean([x["done"] for _, _, x in rolls(r)]), len(rolls(r))))
        for (tag, t), (src, r) in sorted(pts.items()):
            if r.get("hidden") == "settled-v4" or t == 0:
                continue
            if tag not in traces:
                traces[tag] = json.load(gzip.open(f"{SEEN}/analysis/{task}/{tag}_trace.json.gz"))
            tr = traces[tag]
            if tag not in mains:
                mains[tag] = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
            ms = mains[tag]
            g, gp = abs(ms[t, FINGER[0]] - ms[t, FINGER[1]]), abs(ms[t - 1, FINGER[0]] - ms[t - 1, FINGER[1]])
            widening = g - gp > 0.0005
            old_open = g >= OLD_OPEN or widening
            new_open = g >= NEW_OPEN or widening
            if old_open == new_open:
                continue
            touch = [o for o, rr in tr["obj"].items() if rr["touch"][t]]
            ser = sorted(allV[(tag, r["k"])])
            i = [s[0] for s in ser].index(t)
            out.append(dict(task=task, tag=tag, t=t, k=r["k"], src=src, dir=r["_dir"], hidden=r.get("hidden"), gap_mm=round(1000 * g, 1),
                            dgap_mm=round(1000 * (g - gp), 2), touch=touch, V=round(float(ser[i][1]), 3), n=ser[i][2],
                            neighbours=[(s[0], round(float(s[1]), 2)) for s in ser[max(0, i - 2):i + 3]],
                            main_gap_after=[round(1000 * abs(ms[min(t + s, len(ms) - 1), FINGER[0]] - ms[min(t + s, len(ms) - 1), FINGER[1]]), 1) for s in (2, 4, 8)]))
    # the rollouts' finger gap after 2 / 4 / 8 steps (median over the rollouts)
    for o in out:
        r = (coarse(o["task"]) if o["src"] == "coarse" else dense(o["task"]))[(o["tag"], o["t"])]
        gaps = []
        for c, m, x in rolls(r):
            st = np.load(roll_path(r, c, m))["states"]
            gaps.append([1000 * abs(st[min(s, len(st) - 1), FINGER[0]] - st[min(s, len(st) - 1), FINGER[1]]) for s in (2, 4, 8)])
        o["roll_gap_after"] = [round(float(v), 1) for v in np.median(gaps, axis=0)]
    json.dump(out, open(os.path.join(HERE, "grip_rule.json"), "w"), indent=0)
    print(f"{len(out)} current points where the old rule restored 'opening' and the new one would restore 'closing'")
    for o in out:
        print(f"  {o['tag']:32s} t{o['t']:5d} S{o['k']} {o['src']:6s} {o['hidden']:16s} gap {o['gap_mm']} mm (d {o['dgap_mm']}) touch {o['touch']} "
              f"V {o['V']} ({o['n']}) nbrs {o['neighbours']} gap +2/+4/+8 main {o['main_gap_after']} rollouts {o['roll_gap_after']}")


if __name__ == "__main__":
    main()
