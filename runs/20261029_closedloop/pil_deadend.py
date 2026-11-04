"""Does finishing a PackIdenticalLunches stage the wrong way leave the last stage impossible? (2026-10-31)

Stages 0-2 only ask for one more food item in either tupperware; the last stage is the official check (one vegetable
and one meat in each tupperware, no duplicates). Two items of the same kind (vegetable or meat) in one tupperware make
the official check unreachable unless the policy takes one out again: a dead end.
From the critic data (../20261026_critic_data, every step's state saved):
  - which tupperware holds each item, from the bodies' positions (a proxy of OU.check_obj_in_receptacle, which also
    asks for contact): xy distance to the tupperware < XY and height within [Z0, Z1] of it. Checked against the stage:
    at a stage-k decision point exactly k items should be in;
  - at the stage-3 decision points: the stage-3 success rate with a dead-end placement and without;
  - for every rollout that finished stage 1 or 2: how often the finish is a dead end, overall and for the earliest
    finisher among the 8 candidates (the closed loop's rule).
usage: python -I pil_deadend.py   (with the groot python)"""
import os
import collections, functools, sys
import numpy as np, mujoco

RUNS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(RUNS, "20261028_critical"), os.path.join(RUNS, "scripts")]
import data  # noqa: E402
from paths import local  # noqa: E402

TASK = "PackIdenticalLunches"
ITEMS, BOXES = ("vegetable0", "vegetable1", "meat0", "meat1"), ("tupperware0", "tupperware1")
XY, Z0, Z1 = 0.09, -0.03, 0.12


@functools.lru_cache(None)
def adr(tag):
    z = np.load(data.scenes()[tag], allow_pickle=True)
    m = mujoco.MjModel.from_xml_string(local(str(z["xml"])))
    return {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint0")] + 1 for n in ITEMS + BOXES}


def placement(s, a):
    out = {}
    for o in ITEMS:
        p = s[a[o]:a[o] + 3]
        out[o] = next((b for b in BOXES if np.linalg.norm(p[:2] - s[a[b]:a[b] + 2]) < XY and Z0 < p[2] - s[a[b] + 2] < Z1), None)
    return out


def dead(pl):
    """two items of the same kind in one tupperware"""
    return any(sum(1 for o in ITEMS if o.startswith(kind) and pl[o] == b) > 1 for kind in ("vegetable", "meat") for b in BOXES)


def states(r, c, m):
    return np.load(f"{data.roll(r)}/{r['tag']}_t{r['t']}_c{c}_m{m}.npz", allow_pickle=True)["states"]


def main():
    R = [r for ds in data.DATASETS for r in data.rows(TASK, ds).values()]
    # 1. the proxy against the stage: items in at a stage-k decision point
    agree = collections.Counter()
    for r in R:
        n_in = sum(v is not None for v in placement(states(r, 0, 0)[0], adr(r["tag"])).values())
        agree[(r["k"], n_in)] += 1
    print("decision points (stage k, items in by the proxy):", dict(sorted(agree.items())))
    # 2. stage-3 success with and without a dead-end placement
    g = collections.defaultdict(list)
    for r in R:
        if r["k"] == 3:
            pl = placement(states(r, 0, 0)[0], adr(r["tag"]))
            g[dead(pl)].append(data.outcomes(r).mean())
    for d in (False, True):
        if g[d]:
            print(f"stage-3 points, dead end {d!s:5s}: n {len(g[d]):3d}, stage-3 success {100 * np.mean(g[d]):5.1f} %")
    # 3. finishes of stages 1 and 2: how often a dead end; the earliest finisher (the closed loop's pick)
    for k in (1, 2):
        fin, earliest = [], []
        for r in R:
            if r["k"] != k:
                continue
            a = adr(r["tag"]); best = None
            for c in r["cands"]:
                for x in c["rollouts"]:
                    if not x["done"]:
                        continue
                    S = states(r, c["cand"], x["m"])
                    d = dead(placement(S[min(x["steps"], len(S) - 1)], a))
                    fin.append(d)
                    if x["m"] == 0 and (best is None or x["steps"] < best[0]):
                        best = (x["steps"], d)
            if best is not None:
                earliest.append(best[1])
        print(f"stage {k} finishes: {len(fin)} rollouts, dead end {100 * np.mean(fin):5.1f} %;  earliest finisher (continuation 0) at "
              f"{len(earliest)} points, dead end {100 * np.mean(earliest):5.1f} %")


if __name__ == "__main__":
    main()
