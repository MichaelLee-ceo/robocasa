"""split_analysis.layer0: "split points expected if every candidate had the point's P(done)". It plugs the point's sample
mean q into a binomial model, 1 - (1-q^2)^8 - (1-(1-q)^2)^8 + (2q(1-q))^8. But q is estimated from the same 16
outcomes. The null that matches "the candidate does not matter" keeps the point's 16 outcomes and shuffles them over the
8 x 2 slots (a permutation test), which gives a different expectation. Both on the rows in use and on the RUNS rows the
README used. Read-only. usage: python -I check_split_null.py"""
import glob, json, os, sys

R = "/home/michaellee/mclee/robocasa/runs"
DATA = f"{R}/20261026_critic_data"
sys.path.insert(0, f"{R}/20261028_critical")
import numpy as np  # noqa: E402
from effect_by_event import RUNS, points  # noqa: E402

rng = np.random.default_rng(0)


def is_split(y):
    per = y.sum(1)
    return (per == 2).any() and (per == 0).any()


def plug_in(q, n=8):
    a, b, c = q * q, (1 - q) ** 2, 2 * q * (1 - q)
    return 1 - (1 - a) ** n - (1 - b) ** n + c ** n


def perm_prob(s, n=8, reps=20000):
    """P(split) when s successes are spread at random over the n x 2 slots"""
    if s <= 1 or s >= 2 * n - 1:
        return 0.0
    x = np.zeros((reps, 2 * n)); x[:, :s] = 1
    x = rng.permuted(x, axis=1).reshape(reps, n, 2).sum(2)
    return float(((x == 2).any(1) & (x == 0).any(1)).mean())


def main():
    cache = {s: perm_prob(s) for s in range(17)}
    old = []
    for task, d in RUNS.items():
        for f in glob.glob(f"{DATA}/{d}/results_w*.jsonl"):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in r:
                    old.append(dict(task=task, y=np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float)))
    for name, pts in [("rows in use", points()), ("RUNS rows (as in the README)", old)]:
        for task in ["all"] + list(RUNS):
            g = [p for p in pts if task == "all" or p["task"] == task]
            obs = sum(is_split(p["y"]) for p in g)
            e_plug = sum(plug_in(p["y"].mean()) for p in g)
            e_perm = sum(cache[int(p["y"].sum())] for p in g)
            print(f"{name:30s} {task:21s} observed {obs:3d}  expected: plug-in binomial {e_plug:5.1f}, permutation {e_perm:5.1f}")


if __name__ == "__main__":
    main()
