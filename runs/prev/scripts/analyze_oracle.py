"""Aggregate oracle-experiment results, assign heuristic failure labels, write results.json + CSVs."""
import json, glob, os, sys, csv, math, collections

O = sys.argv[1]
rows = []
for p in glob.glob(f"{O}/results_w*.jsonl"):
    for l in open(p):
        rows.append(json.loads(l))
rows = [r for r in rows if "error" not in r]
UNV = json.load(open(f"{O}/tracker_unverified.json")) if os.path.exists(f"{O}/tracker_unverified.json") else {}
for r in rows:
    if r["condition"].startswith("reset"):
        # full task success during a reset rollout implies the subtask was (at least effectively) achieved; e.g. WaffleReheat's
        # success check doesn't require the door to be closed, so the policy can finish the task before the door-close predicate fires.
        r["subtask_success_raw"] = r["subtask_success"]
        r["subtask_success"] = bool(r["subtask_success"] or r["success"])
    k0 = UNV.get(r["task"], {}).get(str(r["demo"]))
    r["tracker_verified"] = not (k0 is not None and (r.get("subtask", 0) >= k0 if r["condition"].startswith("reset") else True))
rows_all = rows
rows = [r for r in rows if r["tracker_verified"] or not r["condition"].startswith("reset")]
JUDGMENT = {"CuttingToolSelection", "BreadSelection", "RecycleBottlesByType", "SeparateFreezerRack"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def label(r):
    """heuristic failure label for a failed rollout (see README for definitions)."""
    if r.get("subtask_success") or (r["condition"] in ("baseline_demoinit", "planner") and r["success"]):
        return None, None
    specs = r["specs"]; hist = r["history"]
    if r["condition"] in ("baseline_demoinit", "planner") and r["reached"] >= r["n_sub"]:
        return "tracker_mismatch", "all subtask predicates met but env._check_success() never true"
    k = r["reached"] if r["condition"] in ("baseline_demoinit", "planner") else r["subtask"]
    k = min(k, len(specs) - 1)
    sp = specs[k]
    t_active = 0
    for st, kk in hist:
        if kk == k - 1:
            t_active = st
    win = [e for e in r["log"] if e["step"] >= t_active]
    tgt = sp.get("obj")
    done_objs = {specs[kk].get("obj") for _, kk in hist}
    grasped_any = set(o for e in win for o in e["grasped"])
    budget = r.get("budget", r["horizon"])
    left = budget - t_active
    sub = f"k{k}:{sp['stage']}"
    if r.get("undone"):
        return "progress/memory", f"{sub} undid completed subtask {r['undone']}"
    if sp["stage"] == "pick":
        wrong = grasped_any - {tgt} - done_objs
        regrasp = (grasped_any & (done_objs - {tgt}))
        if wrong and tgt not in grasped_any:
            return "decision", f"{sub} grasped wrong object {sorted(wrong)} instead of {tgt}"
        if regrasp and tgt not in grasped_any:
            return "progress/memory", f"{sub} re-grasped already-placed {sorted(regrasp)}"
        if tgt in grasped_any:
            return "motor", f"{sub} touched/closed on {tgt} but did not lift it"
        dmin = min([e["d_tgt"] for e in win if e["d_tgt"] is not None] or [9])
        if dmin < 0.10:
            return "motor", f"{sub} reached {tgt} (min dist {dmin:.2f} m) but no grasp"
        if left < 1.5 * sp["demo_len"]:
            return "timeout", f"{sub} only {left} steps left (demo needs {sp['demo_len']})"
        return "progress/stuck", f"{sub} never approached {tgt} (min dist {dmin:.2f} m)"
    if sp["stage"] == "place":
        last = r["log"][-1] if r["log"] else {"grasped": []}
        if tgt in last["grasped"]:
            if left < 1.5 * sp["demo_len"]:
                return "timeout", f"{sub} still holding {tgt}, {left} steps left"
            return "progress/stuck", f"{sub} holding {tgt} but never placed it"
        return "motor_or_decision", f"{sub} released {tgt} but not at a valid location"
    if sp["stage"] == "navigate":
        return "progress/stuck", f"{sub} navigation not completed"
    if left < 1.5 * sp["demo_len"]:
        return "timeout", f"{sub} {sp['skill']} not done, {left} steps left"
    return "motor", f"{sub} fixture skill '{sp['instr']}' not achieved"


for r in rows:
    r["failure_label"], r["failure_detail"] = label(r)
    r["split"] = "composite_unseen"
    r["task_type"] = "judgment" if r["task"] in JUDGMENT else "sequence"
    r.pop("log", None)

json.dump(rows_all, open(f"{O}/results.json", "w"), indent=1, default=str)

# summary per task x condition
summ = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    key = (r["task"], r["condition"])
    s = r.get("subtask_success") if r["condition"].startswith("reset") else r["success"]
    summ[key]["n"] += 1; summ[key]["succ"] += int(bool(s))
    if r["failure_label"]:
        summ[key][r["failure_label"]] += 1
    if not r["condition"].startswith("reset"):
        summ[key]["reached_sum"] += r["reached"]; summ[key]["nsub_sum"] += r["n_sub"]
labels = ["decision", "motor", "motor_or_decision", "progress/stuck", "progress/memory", "timeout", "tracker_mismatch"]
with open(f"{O}/summary_by_task.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "task_type", "condition", "n", "success", "rate", "ci95_lo", "ci95_hi", "mean_frac_subtasks_reached"] + labels)
    for (t, c), v in sorted(summ.items()):
        lo, hi = wilson(v["succ"], v["n"])
        frac = (v["reached_sum"] / v["nsub_sum"]) if v["nsub_sum"] else ""
        w.writerow([t, "judgment" if t in JUDGMENT else "sequence", c, v["n"], v["succ"], round(v["succ"] / v["n"], 3),
                    round(lo, 3), round(hi, 3), round(frac, 3) if frac != "" else ""] + [v[l] for l in labels])
# per-subtask (oracle reset)
ps = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    if r["condition"].startswith("reset"):
        k = (r["task"], r["subtask"], r["sub_stage"], r["condition"])
        ps[k]["n"] += 1; ps[k]["succ"] += int(r["subtask_success"])
with open(f"{O}/summary_reset_by_subtask.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["task", "subtask_idx", "stage", "condition", "n", "success", "rate"])
    for (t, k, st, c), v in sorted(ps.items()):
        w.writerow([t, k, st, c, v["n"], v["succ"], round(v["succ"] / v["n"], 3)])
# overall per condition
tot = collections.defaultdict(lambda: collections.Counter())
for (t, c), v in summ.items():
    for kk, vv in v.items():
        tot[c][kk] += vv
for c, v in sorted(tot.items()):
    lo, hi = wilson(v["succ"], v["n"])
    print(f"{c:18s} {v['succ']}/{v['n']} = {v['succ']/v['n']:.1%} [{lo:.1%},{hi:.1%}] labels:", {l: v[l] for l in labels if v[l]})

# error accumulation, paired per demo: chain_ok(demo) = every subtask succeeded in isolation under oracle reset.
# Under independence, mean(chain_ok) estimates the end-to-end success expected if failures are purely per-subtask;
# e2e << chain suggests accumulated error (drift into states the resets never visit); e2e ~ chain suggests per-subtask
# competence is the bottleneck.
print("\nper-demo chain analysis")
acc = {}
for t in sorted({r["task"] for r in rows}):
    out = dict(demos={})
    for d in sorted({r["demo"] for r in rows if r["task"] == t}):
        e = {}
        for cond in ("baseline_demoinit", "planner"):
            rr = [r for r in rows if r["task"] == t and r["demo"] == d and r["condition"] == cond]
            e[cond] = rr[0]["success"] if rr else None
        for cond in ("reset_task", "reset_sub"):
            rr = sorted([r for r in rows if r["task"] == t and r["demo"] == d and r["condition"] == cond], key=lambda r: r["subtask"])
            n_sub = rr[0]["n_sub"] if rr else 0
            e[cond] = dict(sub=[int(r["subtask_success"]) for r in rr], complete=len(rr) == n_sub,
                           chain_ok=bool(rr) and len(rr) == n_sub and all(r["subtask_success"] for r in rr))
        out["demos"][d] = e
    D = [v for v in out["demos"].values() if v["reset_task"]["complete"] and v["baseline_demoinit"] is not None]
    for key in ("baseline_demoinit", "planner"):
        out[key] = round(sum(bool(v[key]) for v in D) / max(len(D), 1), 3)
    for key in ("reset_task", "reset_sub"):
        DD = [v for v in out["demos"].values() if v[key]["complete"]]
        out[key + "_chain"] = round(sum(v[key]["chain_ok"] for v in DD) / max(len(DD), 1), 3)
        out[key + "_mean_sub"] = round(sum(sum(v[key]["sub"]) for v in DD) / max(sum(len(v[key]["sub"]) for v in DD), 1), 3)
    out["n_paired_demos"] = len(D)
    acc[t] = out
    print(f"{t:22s} n={len(D)} e2e={out['baseline_demoinit']:.2f} planner={out['planner']:.2f} | chain(reset_task)={out['reset_task_chain']:.2f} sub={out['reset_task_mean_sub']:.2f} | chain(reset_sub)={out['reset_sub_chain']:.2f} sub={out['reset_sub_mean_sub']:.2f}")
json.dump(acc, open(f"{O}/error_accumulation.json", "w"), indent=1)

# drift vs independent per-step failure: conditional success of subtask k inside full baseline rollouts (among
# episodes that reached the start of k) vs oracle-reset success of the same subtask (task instruction), same demos.
print("\nconditional per-subtask success: baseline rollout (reached k) vs oracle reset_task")
cond_rows = []
for t in sorted({r["task"] for r in rows}):
    for d in sorted({r["demo"] for r in rows if r["task"] == t}):
        bl = [r for r in rows if r["task"] == t and r["demo"] == d and r["condition"] == "baseline_demoinit"]
        if not bl:
            continue
        bl = bl[0]; done = {kk for _, kk in bl["history"]}
        for r in rows:
            if r["task"] == t and r["demo"] == d and r["condition"] == "reset_task":
                k = r["subtask"]
                reached_k = (k == 0) or ((k - 1) in done)
                if not reached_k:
                    continue
                cond_rows.append(dict(task=t, demo=d, subtask=k, stage=r["sub_stage"],
                                      baseline_ok=bool(k in done or bl["success"]), reset_ok=bool(r["subtask_success"])))
agg = collections.defaultdict(lambda: [0, 0, 0])
for c in cond_rows:
    a = agg[c["task"]]; a[0] += 1; a[1] += c["baseline_ok"]; a[2] += c["reset_ok"]
for t, (n, b, rs) in sorted(agg.items()):
    print(f"{t:22s} n(subtask instances reached in baseline)={n:3d}  baseline-conditional={b/n:.2f}  oracle-reset={rs/n:.2f}")
n = sum(a[0] for a in agg.values()); b = sum(a[1] for a in agg.values()); rs = sum(a[2] for a in agg.values())
both = collections.Counter((c["baseline_ok"], c["reset_ok"]) for c in cond_rows)
print(f"ALL n={n} baseline-conditional={b/n:.3f} oracle-reset={rs/n:.3f}  2x2 (baseline_ok, reset_ok): {dict(both)}")
json.dump(dict(rows=cond_rows, per_task={t: dict(n=a[0], baseline_conditional=a[1] / a[0], reset_task=a[2] / a[0]) for t, a in agg.items()},
               overall=dict(n=n, baseline_conditional=b / n, reset_task=rs / n, table_2x2={str(k): v for k, v in both.items()})),
          open(f"{O}/conditional_subtask_success.json", "w"), indent=1)
