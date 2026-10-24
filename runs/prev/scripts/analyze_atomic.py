"""Summarize the composite->atomic experiment into summary.json (consumed by the HTML report)."""
import json, glob, sys, os, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atomic_steps import TASKS, TEMPLATE_NOTES

A = sys.argv[1]
rows = {}
for p in glob.glob(f"{A}/results_w*.jsonl"):
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            rows[r["tag"]] = r
rows = list(rows.values())


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def rate(xs):
    k, n = sum(xs), len(xs)
    return dict(k=k, n=n, rate=round(k / n, 3) if n else None, ci95=wilson(k, n))


out = dict(tasks={})
for t, steps in TASKS.items():
    R = [r for r in rows if r["task"] == t]
    base = [r for r in R if r["condition"] == "baseline"]
    chain = [r for r in R if r["condition"] == "chain_atomic"]
    any_r = R[0] if R else None
    task_out = dict(
        instruction_example=any_r["task_instr"] if any_r else None,
        atomic_instructions_example=any_r["atomic_instructions"] if any_r else None,
        skills=[s["skill"] for s in steps],
        baseline=rate([r["success"] for r in base]),
        chain_atomic=rate([r["success"] for r in chain]),
        steps=[],
    )
    prod = {"reset_atomic": 1.0, "reset_task": 1.0}
    for k, s in enumerate(steps):
        st = dict(k=k, skill=s["skill"], note=TEMPLATE_NOTES.get(s["skill"]))
        for c in ("reset_atomic", "reset_task"):
            rr = [r for r in R if r["condition"] == c and r.get("step") == k]
            st[c] = rate([r["step_success"] for r in rr])
            if st[c]["rate"] is not None:
                prod[c] *= st[c]["rate"]
        # how often a full run (baseline / chain) got past this step, among runs that reached it
        for c, RR in (("baseline", base), ("chain_atomic", chain)):
            reached = [r for r in RR if r["reached"] >= k or r["success"]]
            passed = [r for r in reached if r["reached"] > k or r["success"]]
            st[f"{c}_conditional"] = dict(k=len(passed), n=len(reached),
                                          rate=round(len(passed) / len(reached), 3) if reached else None)
        task_out["steps"].append(st)
    task_out["product_reset_atomic"] = round(prod["reset_atomic"], 3)
    task_out["product_reset_task"] = round(prod["reset_task"], 3)
    # failure point: first step not completed in failed full runs
    for c, RR in (("baseline", base), ("chain_atomic", chain)):
        fp = collections.Counter(steps[min(r["reached"], len(steps) - 1)]["skill"] + f" (step {min(r['reached'], len(steps)-1)+1})"
                                 for r in RR if not r["success"])
        task_out[f"{c}_failure_point"] = dict(fp)
    task_out["episodes"] = sorted([dict(tag=r["tag"], demo=r["demo"], condition=r["condition"], step=r.get("step"),
                                        success=r["success"], step_success=r.get("step_success"), reached=r["reached"],
                                        steps=r["steps"], budget=r.get("budget", r["horizon"]), instr_log=r["instr_log"],
                                        video=f"{t}/{r['tag']}.mp4") for r in R], key=lambda e: (e["demo"], e["condition"], e["step"] or 0))
    out["tasks"][t] = task_out
out["n_rollouts"] = len(rows)
json.dump(out, open(f"{A}/summary.json", "w"), indent=1)
for t, v in out["tasks"].items():
    print(f"{t:18s} baseline {v['baseline']['k']}/{v['baseline']['n']}  chain {v['chain_atomic']['k']}/{v['chain_atomic']['n']}  "
          f"prod(atomic) {v['product_reset_atomic']}  prod(task) {v['product_reset_task']}")
    for s in v["steps"]:
        print(f"    step{s['k']+1} {s['skill']:28s} reset_atomic {s['reset_atomic']['k']}/{s['reset_atomic']['n']}  reset_task {s['reset_task']['k']}/{s['reset_task']['n']}"
              f"  base_cond {s['baseline_conditional']['k']}/{s['baseline_conditional']['n']}  chain_cond {s['chain_atomic_conditional']['k']}/{s['chain_atomic_conditional']['n']}")
    print("    failure point baseline:", v["baseline_failure_point"], " chain:", v["chain_atomic_failure_point"])
