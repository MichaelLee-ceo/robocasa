"""How far must a critic look ahead? (2026-10-30)

A critic with a perfect world model that rolls each candidate out h chunks (the candidate, then h-1 chunks of the
policy) and a perfect judge of "the stage has finished". Split-half per decision point: per candidate, the fraction of
continuations {0, 1} that finished the stage within 16 h steps; pick the best (ties: the mean over the tied
candidates), score the pick with the full outcome of continuations {2, 3}; and the other way round. h = full is the
split-half ceiling. "finished" is the narrowest judge: one that also reads partial progress could only do better, so
each h gives a lower bound for that horizon.
usage: python -I horizon.py   (with the groot python)"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "20261028_critical")]
import data  # noqa: E402

H = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, None]  # chunks; None = to the end of the budget


def gain(r, h):
    done = np.array([[x["done"] and (h is None or x["steps"] <= 16 * h) for x in c["rollouts"]] for c in r["cands"]], float)
    full = data.outcomes(r)
    g = []
    for A, B in (([0, 1], [2, 3]), ([2, 3], [0, 1])):
        a, b = done[:, A].mean(1), full[:, B].mean(1)
        g.append(b[a == a.max()].mean() - b.mean())
    return 100 * np.mean(g)


def main():
    for ds in ("confirm", "control"):
        print(f"\n=== {ds} ===")
        print(f"{'task':22s} {'n':>3s}  " + "  ".join(f"{('h' + str(h)) if h else 'full':>6s}" for h in H))
        allg = {h: [] for h in H}
        for task in data.TASKS:
            R = list(data.rows(task, ds).values())
            row = []
            for h in H:
                g = [gain(r, h) for r in R]; allg[h] += g; row.append(np.mean(g))
            print(f"{task:22s} {len(R):3d}  " + "  ".join(f"{x:+6.1f}" for x in row))
        print(f"{'all':22s} {len(allg[None]):3d}  " + "  ".join(f"{np.mean(allg[h]):+6.1f}" for h in H))
        st = [x["steps"] for task in data.TASKS for r in data.rows(task, ds).values() for c in r["cands"] for x in c["rollouts"] if x["done"]]
        print("finished rollouts: share finished within h chunks  " +
              "  ".join(f"h{h}:{np.mean(np.array(st) <= 16 * h) * 100:.0f}%" for h in H if h))


if __name__ == "__main__":
    main()
