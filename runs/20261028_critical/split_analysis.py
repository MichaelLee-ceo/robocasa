"""Winners vs losers at the split points (split_points.jsonl), layer by layer.
1. How the losers' rollouts fail stage k (stage_trace verdicts), and when the winners and losers first hold the object
   or touch the fixture.
2. The end of the candidate chunk (frame 16): do the winners end it in a different place from the losers? The mean eef
   distance between a winner and a loser minus the mean within winners and within losers, per point, with a
   permutation test (shuffle win/lose labels within each point). Then which pose features differ in a consistent
   direction (per point mean over winners minus mean over losers; sign test over points).
   Points are split by phase: "carrying" when the winners already hold the object at the decision point, else
   "approaching" (pick-and-place), and "fixture".
3. Where the object is held: the pose at the first hold, winners vs losers that also held it (approaching points).
0. How much of a split is luck: the number of split points expected if every candidate had the point's P(done), and
   the share of the outcome variance that the candidate explains (between-candidate variance / p(1 - p)).
Also lists the points where winners and losers end the chunk furthest apart. Reads split_points.jsonl from $CRIT_OUT;
layer 0 uses the coarse rows in use (data.py). usage: python -I split_analysis.py"""
import collections, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import OUT  # noqa: E402
GRASP_FAIL = {"never_reached", "touched_not_grasped"}
FEATS = ["dist", "height", "yaw_rel", "eef_down", "width", "eef_in_obj_x", "eef_in_obj_y", "eef_in_obj_z"]


def sign_p(k, n):
    """two-sided sign test"""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def feat(p, f):
    if f.startswith("eef_in_obj_"):
        return p["eef_in_obj"]["xyz".index(f[-1])] if "eef_in_obj" in p else None
    return p.get(f)


def load():
    rows = [r for r in map(json.loads, open(os.path.join(OUT, "split_points.jsonl")))]
    bad = [r for r in rows if "error" in r]
    return [r for r in rows if "error" not in r], bad


def layer0():
    from effect_by_event import points
    print("=== 0. how much of a split is luck")
    pts = points()
    for task in ["all", "KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]:
        g = [p for p in pts if task == "all" or p["task"] == task]
        obs = sum(1 for p in g if (p["y"].sum(1) == 2).any() and (p["y"].sum(1) == 0).any())
        exp = 0.0
        for p in g:
            q, n = p["pd"], len(p["y"])
            a, b, c = q * q, (1 - q) ** 2, 2 * q * (1 - q)
            exp += 1 - (1 - a) ** n - (1 - b) ** n + c ** n
        share = sum(p["cb"] for p in g) / sum(p["pd"] * (1 - p["pd"]) for p in g)
        print(f"  {task:21s} split points {obs} observed vs {exp:.0f} expected with no chunk effect; "
              f"the candidate explains {100 * share:.0f} % of the outcome variance")


def layer1(rows):
    print("=== 1. how the losing rollouts fail stage k (verdict of each rollout; a candidate has 2)")
    for task in ["KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]:
        for kind in ["pnp", "fixture"]:
            g = [r for r in rows if r["task"] == task and r["kind"] == kind]
            if not g:
                continue
            c = collections.Counter(v for r in g for cd in r["cands"].values() if not cd["win"] for x in cd["rollouts"] for v in (x["verdict"] or ["(none)"]))
            print(f"  {task} {kind} ({len(g)} points): " + ", ".join(f"{v} {n}" for v, n in c.most_common()))
    print("  first hold (pnp) / first fixture touch, median steps after the decision point; 'never' = rollouts without one")
    for kind, key in [("pnp", "first_hold"), ("fixture", "first_fx_touch")]:
        for win in (True, False):
            v = [x.get(key) for r in rows if r["kind"] == kind for cd in r["cands"].values() if cd["win"] == win for x in cd["rollouts"]]
            have = [x for x in v if x is not None]
            print(f"    {kind:7s} {'winners' if win else 'losers ':7s}: median {np.median(have) if have else float('nan'):5.0f}, never {len(v) - len(have)}/{len(v)}")


def phase(r):
    if r["kind"] != "pnp":
        return "fixture"
    w = [x.get("first_hold") for cd in r["cands"].values() if cd["win"] for x in cd["rollouts"]]
    return "carrying" if w and all(f == 0 for f in w) else "approaching"


def gap(e, win):
    """mean winner-loser eef distance minus the mean within-group distance (mm)"""
    W = [i for i in e if win[i]]; L = [i for i in e if not win[i]]
    bw = [np.linalg.norm(e[a] - e[b]) for a in W for b in L]
    wi = [np.linalg.norm(e[a] - e[b]) for G in (W, L) for n, a in enumerate(G) for b in G[n + 1:]]
    return (np.mean(bw) - np.mean(wi)) * 1000 if bw and wi else None


def layer2(rows):
    print("=== 2. end of the candidate chunk (frame 16)")
    rng = np.random.default_rng(0)
    for ph in ["all", "approaching", "carrying", "fixture"]:
        g = [r for r in rows if ph == "all" or phase(r) == ph]
        obs, null = [], []
        for r in g:
            e = {i: np.array(cd["pose"]["16"]["eef"]) for i, cd in r["cands"].items() if "16" in cd["pose"]}
            win = {i: r["cands"][i]["win"] for i in e}
            x = gap(e, win)
            if x is None:
                continue
            obs.append(x)
            labs = list(win.values())
            null.append([gap(e, dict(zip(e, rng.permutation(labs)))) for _ in range(500)])
        obs, null = np.array(obs), np.array(null, dtype=float)
        p = float((np.nanmean(null, 0) >= obs.mean()).mean()) if len(obs) else float("nan")
        print(f"  {ph:11s} n {len(obs):3d}: winners and losers end the chunk {obs.mean():+.1f} mm further apart than within "
              f"groups (median {np.median(obs):+.1f}; under shuffled labels {np.nanmean(null):+.1f}); permutation p {p:.3f}")
    print("  pose features, per point mean(winners) - mean(losers); sign test over points")
    for scope in ["all", "KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]:
        for f in FEATS:
            d = []
            for r in rows:
                if scope != "all" and r["task"] != scope:
                    continue
                vw = [feat(cd["pose"]["16"], f) for cd in r["cands"].values() if cd["win"] and "16" in cd["pose"]]
                vl = [feat(cd["pose"]["16"], f) for cd in r["cands"].values() if not cd["win"] and "16" in cd["pose"]]
                vw, vl = [x for x in vw if x is not None], [x for x in vl if x is not None]
                if vw and vl:
                    d.append(np.mean(vw) - np.mean(vl))
            d = np.array(d)
            if len(d) >= 8:
                pos, n = int((d > 0).sum()), int((d != 0).sum())
                p = sign_p(pos, n)
                if scope == "all" or p < 0.05:
                    print(f"    {scope:21s} {f:13s} n {len(d):3d}  median diff {np.median(d):+.4f}  winners higher in {pos}/{n}  p {p:.3f}")


def far_apart(rows, top=12):
    print(f"=== 2b. the {top} points where winners and losers end the chunk furthest apart (beyond the within-group spread)")
    out = []
    for r in rows:
        e = {i: np.array(cd["pose"]["16"]["eef"]) for i, cd in r["cands"].items() if "16" in cd["pose"]}
        x = gap(e, {i: r["cands"][i]["win"] for i in e})
        if x is not None:
            out.append((x, r))
    for x, r in sorted(out, key=lambda z: -z[0])[:top]:
        def m(win, key):
            v = [np.array(cd["pose"]["16"][key]) - np.array(cd["pose"]["0"][key]) for cd in r["cands"].values() if cd["win"] == win]
            return np.round(np.mean(v, 0) * 1000).astype(int).tolist()
        verd = collections.Counter(v for cd in r["cands"].values() if not cd["win"] for x2 in cd["rollouts"] for v in (x2["verdict"] or ["-"]))
        print(f"  {x:6.0f} mm  {r['tag']} t {r['t']} S{r['k']} {phase(r):11s} eef move in the chunk (mm) W {m(True, 'eef')} "
              f"L {m(False, 'eef')}; base move W {m(True, 'base')} L {m(False, 'base')}; losers: {dict(verd)}")
    v = np.array([x for x, _ in out])
    print(f"  gap distribution (mm): <5 {np.mean(v < 5):.0%}, 5-20 {np.mean((v >= 5) & (v < 20)):.0%}, 20-50 {np.mean((v >= 20) & (v < 50)):.0%}, "
          f">=50 {np.mean(v >= 50):.0%}")


def layer3(rows):
    print("=== 3. where the object is held: first-hold pose, winners vs losers that held it (approaching points)")
    for scope in ["all", "KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]:
        for f in FEATS:
            d = []
            for r in rows:
                if phase(r) != "approaching" or (scope != "all" and r["task"] != scope):
                    continue
                vw = [feat(x["contact_pose"], f) for cd in r["cands"].values() if cd["win"] for x in cd["rollouts"] if "contact_pose" in x]
                vl = [feat(x["contact_pose"], f) for cd in r["cands"].values() if not cd["win"] for x in cd["rollouts"] if "contact_pose" in x]
                vw, vl = [x for x in vw if x is not None], [x for x in vl if x is not None]
                if vw and vl:
                    d.append(np.mean(vw) - np.mean(vl))
            d = np.array(d)
            if len(d) >= 8:
                pos, n = int((d > 0).sum()), int((d != 0).sum())
                p = sign_p(pos, n)
                if scope == "all" or p < 0.05:
                    print(f"    {scope:21s} {f:13s} n {len(d):3d}  median diff {np.median(d):+.4f}  winners higher in {pos}/{n}  p {p:.3f}")


if __name__ == "__main__":
    rows, bad = load()
    print(f"{len(rows)} split points ({len(bad)} errors): " + ", ".join(f"{t} {sum(r['task'] == t for r in rows)}" for t in ["KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]))
    print("phases: " + ", ".join(f"{ph} {n}" for ph, n in collections.Counter(phase(r) for r in rows).items()))
    layer0(); layer1(rows); layer2(rows); far_apart(rows); layer3(rows)
