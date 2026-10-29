"""Old row vs settled-v5 row of the same decision point: restored ramp, how far the main run's finger gap lies outside
the 8 candidates' range in the first chunk (the anomaly scan's metric; > 3 mm within 4 steps flagged a bad restore),
done rate, error rows, and the per-boundary logs. usage: python -I check_pilot.py <new dir> [<new dir> ...]"""
import glob, json, os, sys
import numpy as np

D = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"
MAIN = "/home/michaellee/mclee/robocasa/runs/20261025_seen480/run"
F1, F2 = 12, 13  # finger joints in the flattened state (1 + qpos index; the robot comes first in every task)


def rows(d):
    out = []
    for p in sorted(glob.glob(f"{d}/results_w*.jsonl")):
        for l in open(p):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def find(d, tag, t):
    for r in rows(f"{D}/{d}"):
        if r.get("tag") == tag and r.get("t") == t and "error" not in r:
            return r


def gaps(d, tag, t, N):
    G = []
    for i in range(N):
        p = f"{D}/{d}/roll/{tag}_t{t}_c{i}_m0.npz" if not d.startswith("/") else f"{d}/roll/{tag}_t{t}_c{i}_m0.npz"
        s = np.load(p)["states"][:17]
        G.append(np.abs(s[:, F1] - s[:, F2]))
    n = min(len(g) for g in G)
    return np.array([g[:n] for g in G])


def replaced():
    """(tag, t) -> dir the row replaces, from the jobs files in this folder (rows from before "replaces" was logged)."""
    m = {}
    for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs*.json")):
        for j in json.load(open(p)):
            if "replaces" in j:
                m[(j["tag"], j["t"], j.get("out"))] = j["replaces"]
    return m


def main():
    rep = replaced()
    for nd in sys.argv[1:]:
        for r in rows(nd):
            if "error" in r:
                print("ERROR ROW", r["tag"], r["t"], r["error"][:300]); continue
            task, tag, t = r["task"], r["tag"], r["t"]
            old_dir = r.get("replaces") or rep.get((tag, t, nd)) or rep.get((tag, t, os.path.abspath(nd)))
            old = find(old_dir, tag, t) if old_dir else None
            ep = np.load(f"{MAIN}/{task}/{tag}_steps.npz")["states"]
            gm = np.abs(ep[t:t + 17, F1] - ep[t:t + 17, F2])
            line = [f"{task[:3]} {tag.split('_')[1]} t{t} k{r['k']}", f"ramp v5 {r['restore']['ramp']:+.1f} (gap {r['restore']['gap_mm']} mm, "
                    f"dgap {r['restore'].get('dgap_mm')}, held {r['restore']['held']:d} obj {r['restore']['obj']:d} fix {r['restore']['fixture']:d})"]
            for name, d_, row in (("old", old_dir, old), ("new", nd, r)):
                if row is None:
                    line.append(f"{name}: -"); continue
                G = gaps(d_, tag, t, row["N"])
                n = min(G.shape[1], len(gm))
                out = [max(gm[s] - G[:, s].max(), G[:, s].min() - gm[s], 0) * 1000 for s in (1, 2, 4, 8, 16) if s < n]
                done = np.mean([ro["done"] for c in row["cands"] for ro in c["rollouts"]])
                line.append(f"{name} [{row.get('hidden')}]: main outside cands (mm) @1,2,4,8,16 = {', '.join(f'{x:.1f}' for x in out)}; "
                            f"cand gap@4 {G[:, min(4, n - 1)].min() * 1000:.0f}-{G[:, min(4, n - 1)].max() * 1000:.0f} vs main {gm[min(4, n - 1)] * 1000:.0f}; done {done:.2f}")
            nb = [len(ro.get("bounds", [])) for c in r["cands"] for ro in c["rollouts"]]
            line.append(f"bounds logged per rollout {min(nb)}-{max(nb)}; steps {[ro['steps'] for c in r['cands'] for ro in c['rollouts']][:8]}")
            print("\n  ".join(line))


if __name__ == "__main__":
    main()
