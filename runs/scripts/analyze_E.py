"""Experiment E: the same protocols served by the Atomic-Seen checkpoint (conditions *_ackpt), paired with the
composite_seen runs (same seeds / same demos)."""
import json, glob, os, sys
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomic_steps import TASKS

RUNS = os.path.expanduser("~/Desktop/robocasa/runs")
SD = f"{RUNS}/20261012_atomic_seeded"; AD = f"{RUNS}/20261012_atomic_decomp_seen"
ATOMIC_SEEN = {"CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug", "NavigateKitchen", "OpenCabinet",
               "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
               "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack", "TurnOffStove", "TurnOnElectricKettle",
               "TurnOnMicrowave", "TurnOnSinkFaucet"}  # robocasa/utils/dataset_registry.py TARGET_TASKS["atomic_seen"]


def load(d):
    R = {}
    for p in glob.glob(f"{d}/results_w*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r:
                R[r["tag"]] = r
    return list(R.values())


RS, RA = load(SD), load(AD)


def mc(a, b):
    ks = sorted(set(a) & set(b))
    x = sum(a[s] and not b[s] for s in ks); y = sum(b[s] and not a[s] for s in ks)
    return dict(n=len(ks), a=sum(a[s] for s in ks), b=sum(b[s] for s in ks), a_only=x, b_only=y,
                p=round(binomtest(x, x + y, .5).pvalue, 4) if x + y else None)


def per_step(rows, nst):
    out = []
    for k in range(nst):
        reached = [r for r in rows if r["reached"] >= k or r["success"]]
        out.append([sum(r["reached"] > k or r["success"] for r in reached), len(reached)])
    return out


out = {}
for t, steps in TASKS.items():
    T = dict(skills=[s["skill"] for s in steps], in_atomic_seen=[s["skill"] in ATOMIC_SEEN for s in steps])
    by = {c: {r["seed"]: r for r in RS if r["task"] == t and r["condition"] == c}
          for c in ("baseline_seed", "chain_seed", "chain_seed_ackpt", "baseline_seed_ackpt")}
    T["fresh"] = {c: dict(k=sum(r["success"] for r in v.values()), n=len(v), steps=per_step(list(v.values()), len(steps)))
                  for c, v in by.items()}
    succ = {c: {s: r["success"] for s, r in v.items()} for c, v in by.items()}
    T["fresh_tests"] = {"chain_ackpt vs chain": mc(succ["chain_seed_ackpt"], succ["chain_seed"]),
                        "chain_ackpt vs full": mc(succ["chain_seed_ackpt"], succ["baseline_seed"]),
                        "full_ackpt vs full": mc(succ["baseline_seed_ackpt"], succ["baseline_seed"])}
    # step k from the correct start, fresh scenes (states saved from the composite_seen full run, experiment D)
    T["fresh_restore"] = {}
    for k in range(len(steps)):
        g = lambda c: {r["seed"]: r["step_success"] for r in RS if r["task"] == t and r["condition"] == c and r.get("step") == k}
        a, c, f = g("reset_seed_atomic_ackpt"), g("reset_seed_atomic"), g("reset_seed_task")
        if a:
            T["fresh_restore"][k] = dict(ackpt_vs_cs_atomic=mc(a, c), ackpt_vs_cs_full=mc(a, f))
    # step k from the demo start
    T["demo_restore"] = {}
    for k in range(len(steps)):
        a = {r["demo"]: r["step_success"] for r in RS if r["task"] == t and r["condition"] == "reset_atomic_ackpt" and r["step"] == k}
        c = {r["demo"]: r["step_success"] for r in RA if r["task"] == t and r["condition"] == "reset_atomic" and r["step"] == k}
        f = {r["demo"]: r["step_success"] for r in RA if r["task"] == t and r["condition"] == "reset_task" and r["step"] == k}
        T["demo_restore"][k] = dict(ackpt_vs_cs_atomic=mc(a, c), ackpt_vs_cs_full=mc(a, f))
    out[t] = T
json.dump(out, open(f"{SD}/summary_E.json", "w"), ensure_ascii=False, indent=1)

for t, T in out.items():
    F = T["fresh"]
    print(f"== {t}   fresh full(cs) {F['baseline_seed']['k']}/{F['baseline_seed']['n']}  chain(cs) {F['chain_seed']['k']}/{F['chain_seed']['n']}"
          f"  chain(AS) {F['chain_seed_ackpt']['k']}/{F['chain_seed_ackpt']['n']}  full(AS) {F['baseline_seed_ackpt']['k']}/{F['baseline_seed_ackpt']['n']}")
    print("   tests", {k: (v["a_only"], v["b_only"], v["p"]) for k, v in T["fresh_tests"].items()})
    for k, sk in enumerate(T["skills"]):
        fs = lambda c: "%d/%d" % tuple(F[c]["steps"][k])
        line = f"   step {k+1} {sk:28s} {'AS' if T['in_atomic_seen'][k] else '--'}  chain: cs {fs('chain_seed')} AS {fs('chain_seed_ackpt')}  full: cs {fs('baseline_seed')} AS {fs('baseline_seed_ackpt')}"
        d = T["demo_restore"][k]
        line += f" | demo-start: AS {d['ackpt_vs_cs_atomic']['a']}/{d['ackpt_vs_cs_atomic']['n']} cs-atomic {d['ackpt_vs_cs_atomic']['b']} (p={d['ackpt_vs_cs_atomic']['p']}) cs-full {d['ackpt_vs_cs_full']['b']}"
        if k in T["fresh_restore"]:
            e = T["fresh_restore"][k]
            line += f" | fresh-start: AS {e['ackpt_vs_cs_atomic']['a']}/{e['ackpt_vs_cs_atomic']['n']} cs-atomic {e['ackpt_vs_cs_atomic']['b']} cs-full {e['ackpt_vs_cs_full']['b']}"
        print(line)
