"""Stratified sample of rollouts for label_check.py: per task x stage x done, 10 rollouts (5 from the current coarse
rows, 5 from the dense rows when there are enough), each from a different decision point; plus edge cases:
success at a non-last stage, success with reached == k (transient success), prev_ok False (SteamInMicrowave S1/S2).
Writes sample.json (one job per decision point). usage: python make_sample.py"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, NSTAGES, TASKS3, coarse, dense, rolls, scenes  # noqa: E402


def main():
    rng = random.Random(20261029)
    sc = scenes()
    picks = {}  # (dir, tag, t) -> (row, [(c, m, label, why)])

    def add(r, c, m, x, why):
        key = (r["_dir"], r["tag"], r["t"])
        if key not in picks:
            picks[key] = (r, [])
        if all((c, m) != (cc, mm) for cc, mm, _, _ in picks[key][1]):
            picks[key][1].append((c, m, {f: x[f] for f in ("done", "prev_ok", "steps", "success", "reached", "progress")}, why))

    for task in TASKS3:
        rows = {"coarse": list(coarse(task).values()), "dense": list(dense(task).values())}
        for k in range(NSTAGES[task]):
            for dn in (True, False):
                cells = {s: [(r, c, m, x) for r in rows[s] if r["k"] == k for c, m, x in rolls(r) if x["done"] == dn] for s in rows}
                want = {"coarse": 5, "dense": 5}
                if len({(r["tag"], r["t"]) for r, *_ in cells["dense"]}) < 5:
                    want = {"coarse": 10 - len({(r["tag"], r["t"]) for r, *_ in cells["dense"]}), "dense": 5}
                for s in ("coarse", "dense"):
                    by_pt = {}
                    for r, c, m, x in cells[s]:
                        by_pt.setdefault((r["tag"], r["t"]), []).append((r, c, m, x))
                    keys = sorted(by_pt)
                    rng.shuffle(keys)
                    for key in keys[:want[s]]:
                        r, c, m, x = rng.choice(by_pt[key])
                        add(r, c, m, x, f"strat {task} S{k} done={dn} {s}")
        # edge cases
        allr = [(r, c, m, x) for s in rows for r in rows[s] for c, m, x in rolls(r)]
        last = NSTAGES[task] - 1
        edge = {"success_nonlast": [z for z in allr if z[3]["success"] and z[0]["k"] < last],
                "success_transient": [z for z in allr if z[3]["success"] and z[3]["reached"] == z[0]["k"]],
                "prev_ok_false": [z for z in allr if not z[3]["prev_ok"]],
                "reached_2": [z for z in allr if z[3]["reached"] >= z[0]["k"] + 2]}
        for why, zs in edge.items():
            rng.shuffle(zs)
            for r, c, m, x in zs[:4]:
                add(r, c, m, x, f"edge {why}")
    jobs = []
    for (d, tag, t), (r, rl) in picks.items():
        jobs.append([r["task"], sc[tag], d, tag, t, r["k"], r["L"], r["py_state"], r["k_replay"], [z[:3] for z in rl]])
    json.dump(jobs, open(os.path.join(HERE, "sample.json"), "w"))
    whys = {}
    for _, (_, rl) in picks.items():
        for z in rl:
            whys[z[3].split(" done")[0] if z[3].startswith("strat") else z[3]] = whys.get(z[3].split(" done")[0] if z[3].startswith("strat") else z[3], 0) + 1
    print(f"{sum(len(rl) for _, rl in picks.values())} rollouts at {len(jobs)} points")
    for w, n in sorted(whys.items()):
        print(f"   {w}: {n}")
    json.dump({f"{d}|{tag}|{t}": [[c, m, why] for c, m, _, why in rl] for (d, tag, t), (_, rl) in picks.items()},
              open(os.path.join(HERE, "sample_why.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
