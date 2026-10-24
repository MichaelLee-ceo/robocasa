"""Detailed flat exports for diagnosis: per-episode (baseline/planner), per-subtask-instance (reset), aggregates."""
import json, csv, sys, collections, os
O = sys.argv[1]
rs = json.load(open(f"{O}/results.json"))
vid = lambda r: f"{O}/{r['task']}/{r['tag']}.mp4"

with open(f"{O}/episodes_baseline_planner.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "task_type", "demo", "condition", "success", "steps", "horizon", "subtasks_completed", "n_subtasks",
                "last_completed_subtask", "first_failed_subtask", "completion_steps", "failure_label", "failure_detail", "video"])
    for r in sorted([r for r in rs if r["condition"] in ("baseline_demoinit", "planner")], key=lambda r: (r["task"], r["demo"], r["condition"])):
        sp = r["specs"]; k = r["reached"]
        w.writerow([r["task"], r["task_type"], r["demo"], r["condition"], int(r["success"]), r["steps"], r["horizon"], k, r["n_sub"],
                    f"k{k-1}:{sp[k-1]['instr']}" if k > 0 else "", f"k{k}:{sp[k]['instr']}" if k < len(sp) and not r["success"] else "",
                    ";".join(f"k{kk}@{st}" for st, kk in r["history"]), r["failure_label"] or "", r["failure_detail"] or "", vid(r)])

inst = collections.defaultdict(dict)
for r in rs:
    if r["condition"].startswith("reset"):
        inst[(r["task"], r["demo"], r["subtask"])][r["condition"]] = r
with open(f"{O}/reset_subtask_instances.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "demo", "subtask_idx", "stage", "skill", "subtask_instruction", "demo_segment_len", "budget", "tracker_verified",
                "reset_task_ok", "reset_task_steps", "reset_task_label", "reset_task_detail", "reset_sub_ok", "reset_sub_steps", "reset_sub_label", "reset_sub_detail"])
    for (t, d, k), v in sorted(inst.items()):
        a, b = v.get("reset_task"), v.get("reset_sub"); x = a or b
        w.writerow([t, d, k, x["sub_stage"], x["sub_skill"], x["sub_instr"], x["specs"][k]["demo_len"], x["budget"], int(x["tracker_verified"]),
                    int(a["subtask_success"]) if a else "", a["steps"] if a else "", (a or {}).get("failure_label") or "", (a or {}).get("failure_detail") or "",
                    int(b["subtask_success"]) if b else "", b["steps"] if b else "", (b or {}).get("failure_label") or "", (b or {}).get("failure_detail") or ""])

agg = collections.defaultdict(lambda: dict(n=0, task_ok=0, sub_ok=0, budgets=[], instrs=collections.Counter()))
for (t, d, k), v in inst.items():
    a, b = v.get("reset_task"), v.get("reset_sub")
    if not (a and b) or not a["tracker_verified"]:
        continue
    g = agg[(t, k, a["sub_stage"])]
    g["n"] += 1; g["task_ok"] += int(a["subtask_success"]); g["sub_ok"] += int(b["subtask_success"]); g["budgets"].append(a["budget"]); g["instrs"][a["sub_instr"]] += 1
with open(f"{O}/summary_reset_by_subtask.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "subtask_idx", "stage", "n", "reset_task_success", "reset_sub_success", "budget_min", "budget_max", "subtask_instructions(count)"])
    for (t, k, st), g in sorted(agg.items()):
        w.writerow([t, k, st, g["n"], g["task_ok"], g["sub_ok"], min(g["budgets"]), max(g["budgets"]),
                    " || ".join(f"{i} ({c})" for i, c in g["instrs"].most_common())])
print("wrote episodes_baseline_planner.csv, reset_subtask_instances.csv, summary_reset_by_subtask.csv")
