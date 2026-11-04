"""Why does a critic's pick gain nothing? (2026-10-30)

Layer by layer, at the critical (confirm) points:
  1. the ceiling: how much a perfect critic could gain. Split-half: pick the best candidate on 2 of the 4 continuations,
     score it on the other 2 (both ways), so the luck of the 4 continuations is not counted (the "+25" of
     ../20261028_critical is the same measure on all its confirm points);
  2. does the commanded path carry the signal? One-feature critics (pick the candidate with the largest, or smallest,
     value of one feature of its commanded path): their gain, and the within-point Spearman of the feature with the
     rates. The sign is chosen afterwards, so the better of the two is optimistic;
  3. what does the critic follow? Spearman of its ranking with each feature;
  4. the critic's gain by its own confidence.
usage: python -I analyze_blind.py <answers dir>   (with the groot python)"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "20261028_critical")]
import data  # noqa: E402
from commanded import commanded, first_chunk  # noqa: E402


def angle(Ra, Rb):
    return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def features(cands):
    ends = np.array([c["pos"][-1] for c in cands]); med = np.median(ends, 0)
    F = {}
    F["path length"] = [float(np.linalg.norm(np.diff(c["pos"], axis=0), axis=1).sum()) for c in cands]
    F["end displacement"] = [float(np.linalg.norm(c["pos"][-1] - c["pos"][0])) for c in cands]
    F["lift (dz)"] = [float(c["pos"][-1, 2] - c["pos"][0, 2]) for c in cands]
    F["rotation"] = [angle(c["rot"][0], c["rot"][-1]) for c in cands]
    F["steps commanded closed"] = [float(c["close"].sum()) for c in cands]
    F["closed at the end"] = [float(c["close"][-1]) for c in cands]
    F["base steps"] = [float(c["base_mode"].sum()) for c in cands]
    F["typicality (-dist to median end)"] = [-float(np.linalg.norm(e - med)) for e in ends]
    return F


def main():
    d = sys.argv[1]
    key, truth = json.load(open(f"{HERE}/key.json")), json.load(open(f"{HERE}/truth.json"))
    pts = []
    for pid in sorted(key):
        k = key[pid]
        r = data.rows(k["task"], k["dataset"])[(k["tag"], k["t"])]
        O = data.outcomes(r)  # 8 x 4
        assert np.allclose(O.mean(1), truth[pid]), pid
        s = np.load(data.snap(r))
        cands = [commanded(s, first_chunk(f"{data.roll(r)}/{r['tag']}_t{r['t']}_c{i}_m0.npz")[0]) for i in range(8)]
        a = json.load(open(f"{d}/{pid}.json"))
        pts.append(dict(pid=pid, **k, O=O, rates=O.mean(1), F=features(cands), pick=a["pick"], rank=a["ranking"],
                        conf=a.get("confidence")))

    for ds in ("confirm", "control"):
        P = [p for p in pts if p["dataset"] == ds]
        print(f"\n=== {ds}: {len(P)} points ===")
        # 1. ceiling, split-half
        sh = []
        for p in P:
            for A, B in (([0, 1], [2, 3]), ([2, 3], [0, 1])):
                ra, rb = p["O"][:, A].mean(1), p["O"][:, B].mean(1)
                best = np.flatnonzero(ra == ra.max())  # ties: the mean over the tied candidates
                sh.append(100 * (rb[best].mean() - rb.mean()))
        crit = [100 * (p["rates"][p["pick"]] - p["rates"].mean()) for p in P]
        print(f"ceiling, split-half {np.mean(sh):+5.1f} pts   in-sample best-of-8 {np.mean([100 * (p['rates'].max() - p['rates'].mean()) for p in P]):5.1f}"
              f"   critic {np.mean(crit):+5.1f}")
        # 2. one-feature critics, 3. what the critic follows
        print(f"{'feature':34s} {'gain max':>9s} {'gain min':>9s} {'rho(feat, rate)':>16s} {'rho(critic, feat)':>18s}")
        for f in P[0]["F"]:
            gmax, gmin, rr, rc = [], [], [], []
            for p in P:
                v, rt = np.array(p["F"][f]), p["rates"]
                gmax.append(100 * (rt[v == v.max()].mean() - rt.mean())); gmin.append(100 * (rt[v == v.min()].mean() - rt.mean()))
                if v.max() > v.min():  # Spearman with ties averaged (the features and the rates have ties)
                    if rt.max() > rt.min():
                        rr.append(np.corrcoef(_rk(v), _rk(rt))[0, 1])
                    rc.append(np.corrcoef(_rk(v), _rk(-np.argsort(p["rank"]).astype(float)))[0, 1])
            print(f"{f:34s} {np.mean(gmax):+9.1f} {np.mean(gmin):+9.1f} {np.mean(rr) if rr else float('nan'):+11.2f} (n {len(rr):2d})"
                  f" {np.mean(rc) if rc else float('nan'):+12.2f} (n {len(rc):2d})")
        # 4. by confidence
        for c in ("low", "medium", "high"):
            g = [100 * (p["rates"][p["pick"]] - p["rates"].mean()) for p in P if p["conf"] == c]
            if g:
                print(f"confidence {c:6s} n {len(g):2d}  gain {np.mean(g):+5.1f}")
    print("\n=== confirm points ===")
    for p in sorted([p for p in pts if p["dataset"] == "confirm"], key=lambda p: (p["task"], p["stage"])):
        rt = p["rates"]
        print(f"{p['pid']} {p['task'][:5]} k{p['stage']} {p['tag'].split('_')[1]:4s} t{p['t']:5d}  rates {' '.join(f'{x:.2f}' for x in rt)}"
              f"  pick #{p['pick']} ({rt[p['pick']]:.2f}, {p['conf']})  best {list(np.flatnonzero(rt == rt.max()))}")


def _rk(v):
    """ranks with ties averaged"""
    v = np.asarray(v, float); o = np.argsort(v, kind="stable"); r = np.empty(len(v)); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]: j += 1
        r[o[i:j + 1]] = (i + j) / 2; i = j + 1
    return r


if __name__ == "__main__":
    main()
