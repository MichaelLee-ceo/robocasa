"""Gripper ramp restore check over every current rollout (coarse + dense): do the fingers move against the commanded side
in the first steps after the restore, more often than at a later chunk boundary of the same rollouts (where the ramp
is the true, continuing one)?
  window W = steps [s, s + 4) with a constant gripper command (gripper_close 0 = open, 1 = close);
  against = command open and the finger gap shrinks by > 5 mm over the 4 steps, or command close and it grows by > 2 mm
  while the fingers are not already closed on something (gap > 10 mm at s).
First window s = 0 (just after restore + settle) vs control windows s = 16 and s = 32 (chunk boundaries inside the
same rollout). Also, per rollout: states[0] == the main run's state at t (max abs diff), and whether its first step is in
base mode (bug 2 hit rows 'settled' whose first step is in base mode). Writes grip_against.json.
usage: python grip_against.py"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, SEEN, TASKS3, coarse, dense, rolls, roll_path  # noqa: E402

F = (12, 13)


def gap(s):
    return 1000 * abs(s[F[0]] - s[F[1]])


def against(st, act, s):
    """None if the window is not usable, else (against: bool, command, gap change)"""
    c = s // 16
    if c >= len(act) or s + 4 >= len(st):
        return None
    cmd = act[c, :4, 0]
    if not (np.all(cmd == cmd[0])):
        return None
    g0, g4 = gap(st[s]), gap(st[s + 4])
    if cmd[0] < 0.5:
        return (g4 < g0 - 5, "open", round(g4 - g0, 1))
    if g0 <= 10:
        return None
    return (g4 > g0 + 2, "close", round(g4 - g0, 1))


def main():
    out, tot = [], {}
    start_bad, base_start = [], []
    for task in TASKS3:
        mains = {}
        for src, rows in (("coarse", coarse(task)), ("dense", dense(task))):
            for (tag, t), r in rows.items():
                hits = {0: [], 16: [], 32: []}
                if tag not in mains:
                    mains[tag] = np.load(f"{SEEN}/run/{task}/{tag}_steps.npz")["states"]
                for c, m, x in rolls(r):
                    z = np.load(roll_path(r, c, m))
                    st, act = z["states"], z["action.gripper_close"]
                    d0 = float(np.abs(st[0] - mains[tag][t]).max())
                    if d0 > 0:
                        start_bad.append((task, tag, t, c, m, d0))
                    if z["action.control_mode"][0, 0, 0] >= 0.5:
                        base_start.append((task, tag, t, c, m, r.get("hidden"), r["_dir"]))
                    for s in hits:
                        a = against(st, act, s)
                        if a is not None:
                            hits[s].append(a)
                for s, h in hits.items():
                    key = (task, r.get("hidden"), s)
                    n, k = tot.get(key, (0, 0))
                    tot[key] = (n + len(h), k + sum(a[0] for a in h))
                n0 = len(hits[0]); k0 = sum(a[0] for a in hits[0])
                if k0:
                    out.append(dict(task=task, tag=tag, t=t, k=r["k"], src=src, dir=r["_dir"], hidden=r.get("hidden"), n=n0, against=k0,
                                    cmds=sorted({a[1] for a in hits[0] if a[0]}), dgap=[a[2] for a in hits[0] if a[0]][:8],
                                    V=float(np.mean([x["done"] for _, _, x in rolls(r)])),
                                    control_against=sum(a[0] for a in hits[16] + hits[32]), control_n=len(hits[16] + hits[32])))
    json.dump(dict(points=out, totals={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in tot.items()}, start_bad=start_bad, base_start=base_start),
              open(os.path.join(HERE, "grip_against.json"), "w"), indent=0)
    print(f"rollouts whose start state differs from the main run's state at t: {len(start_bad)} {start_bad[:5]}")
    import collections
    print("rollouts whose first step is in base mode, by code version:", dict(collections.Counter(b[5] for b in base_start)))
    print("   of them in rows 'settled' (bug 2 not fixed there):", [b for b in base_start if b[5] == "settled"][:20])
    print("rate of 'fingers move against the command' per window (task, hidden, window start): against / usable windows")
    for key in sorted(tot, key=lambda k: (k[0], str(k[1]), k[2])):
        n, k = tot[key]
        print(f"   {key[0]:21s} {str(key[1]):16s} s={key[2]:2d}: {k:4d} / {n:5d} = {k / max(n, 1):.3f}")
    print(f"points with any first-window 'against' rollout: {len(out)}")
    for o in sorted(out, key=lambda o: -o["against"] / o["n"]):
        if o["against"] / o["n"] >= 0.25:
            print(f"   {o['tag']:32s} t{o['t']:5d} S{o['k']} {o['src']:6s} {o['hidden']:16s} against {o['against']}/{o['n']} {o['cmds']} dgap {o['dgap']} "
                  f"V {o['V']:.2f}  (control windows: {o['control_against']}/{o['control_n']})")


if __name__ == "__main__":
    main()
