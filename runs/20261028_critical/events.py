"""Physical events (source B of the critical-moment search) in the seed-0 main runs, from the saved per-step traces
(../20261025_seen480/analysis/<task>/<tag>_trace.json.gz, made by ../scripts/stage_trace.py). No simulation.

Each event: step t (index into the episode's saved states, as the decision points' t), the stage k in progress at t
(from the main run's history), its type and the object or fixture:
  pick-and-place stages, per object of the stage:
    reach    eef within 10 cm of the object for the first time in the stage
    touch    gripper starts touching the object (after not touching it for a chunk; the same for at and fx_touch)
    hold     a hold starts (touching while grasped or lifted >= 2 cm; stage_trace._holds)
    lift     the object first rises >= 2 cm above its height at the stage start, during a hold
    at       the object reaches the stage's target (official predicate, without the release part)
    release  a hold ends
  fixture stages:
    fx_touch gripper starts touching the fixture
    fx_move  the fixture's official value first moves >= 0.02 from its value at the stage start
    fx_done  the fixture's done predicate turns true
  any stage:
    close    the fingers close past 6 cm (grip width falls through 0.06 m from above)
    open     the fingers open past 6 cm
Writes events.jsonl (one row per episode). usage: python events.py"""
import gzip, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from stage_trace import STAGE_SPECS, _holds

TASKS = ["KettleBoiling", "SteamInMicrowave", "PackIdenticalLunches"]
SEEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "20261025_seen480")


def rises(mask, a, b, gap=16):
    """steps in [a, b) where mask turns true after being false for the previous `gap` steps (one chunk): contacts and
    the at-target predicates flicker at their thresholds."""
    return [f for f in range(a, b) if mask[f] and not any(mask[max(0, f - gap):f])]


def episode_events(task, tr, hist):
    T = len(tr["grip_width"])
    ends = [t for t, _ in hist] + [T] * (len(STAGE_SPECS[task]) - len(hist))
    out = []
    for k, sp in enumerate(STAGE_SPECS[task]):
        a = 0 if k == 0 else ends[k - 1]
        b = min(ends[k] + 1, T)
        if a >= T:
            break
        ev = lambda t, typ, what: out.append(dict(t=t, k=k, type=typ, what=what))
        if sp["kind"] == "pnp":
            for o in sp["objs"]:
                r = tr["obj"][o]
                near = [d < 0.10 for d in r["eef_dist"]]
                f = next((f for f in range(a, b) if near[f]), None)
                if f is not None:
                    ev(f, "reach", o)
                for f in rises(r["touch"], a, b):
                    ev(f, "touch", o)
                z0, lifted = r["pos"][a][2], False
                for s, e in _holds(r, a, b):
                    ev(s, "hold", o)
                    if e < b:
                        ev(e, "release", o)
                    f = next((f for f in range(s, e) if r["pos"][f][2] > z0 + 0.02), None)
                    if f is not None and not lifted:
                        ev(f, "lift", o); lifted = True
                for f in rises([x.get(o, False) for x in tr["at"][k]], a, b):
                    ev(f, "at", o)
        elif sp["kind"] == "fixture":
            fx = sp["fixture"]
            for f in rises(tr["fx_touch"][fx], a, b):
                ev(f, "fx_touch", fx)
            v = tr["fx_value"][k]
            f = next((f for f in range(a, b) if abs(v[f] - v[a]) >= 0.02), None)
            if f is not None:
                ev(f, "fx_move", fx)
            for f in rises(tr["fx_done"][k], a, b):
                ev(f, "fx_done", fx)
        g = tr["grip_width"]
        for f in range(max(a, 1), b):
            if g[f - 1] >= 0.06 > g[f]:
                ev(f, "close", "gripper")
            elif g[f - 1] < 0.06 <= g[f]:
                ev(f, "open", "gripper")
    return sorted(out, key=lambda e: (e["t"], e["type"]))


def main():
    main_rows = {}
    for p in sorted(os.listdir(f"{SEEN}/run")):
        if p.startswith("results_w"):
            for r in map(json.loads, open(f"{SEEN}/run/{p}")):
                main_rows[r["tag"]] = r
    n = 0
    with open("events.jsonl", "w") as f:
        for task in TASKS:
            for s in range(30):
                tag = f"{task}_s{s}_single"
                tr = json.load(gzip.open(f"{SEEN}/analysis/{task}/{tag}_trace.json.gz"))
                m = main_rows[tag]
                evs = episode_events(task, tr, m["history"])
                f.write(json.dumps(dict(tag=tag, task=task, T=len(tr["grip_width"]), success=m["success"],
                                        history=m["history"], events=evs)) + "\n")
                n += len(evs)
    print(f"wrote events.jsonl: 90 episodes, {n} events")


if __name__ == "__main__":
    main()
