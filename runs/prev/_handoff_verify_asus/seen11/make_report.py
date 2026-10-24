"""Combine audit11.json + val_<task>.json into report.json (no simulator needed)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "audit11.json")))
SKILLS = {  # mirror of atomic_steps_seen11.TASKS skills/seg (kept here so this script needs no robocasa import)
}
out = {}
for t, a in A.items():
    V = json.load(open(os.path.join(HERE, f"val_{t}.json")))
    main = [r for r in V if not r["extra"]]
    skills = main[0]["step_skills"]
    n = len(skills)
    per_step = []
    for i in range(n):
        p = sum(1 for r in main if r["n_passed"] > i)
        d = [r["pass_minus_seg_end"][i] for r in main if len(r["pass_minus_seg_end"]) > i and r["annotation_matches"]]
        per_step.append(dict(step=i, skill=skills[i], passed=p, of=len(main),
                             false_at_seg_start=sum(1 for r in main if r["false_at_seg_start"][i]),
                             first_pass_minus_annotated_end=dict(min=min(d) if d else None, max=max(d) if d else None,
                                                                 median=sorted(d)[len(d) // 2] if d else None)))
    demos = []
    for r in V:
        demos.append(dict(ep=r["ep"], extra_nonmajority_demo=r["extra"], annotated_steps=r["annotated_steps"],
                          annotation_matches_steps=r["annotation_matches"], all_passed=r["all_passed"],
                          first_pass_frame=r["first_pass_frame"], annotated_step_start=r["seg_start"],
                          annotated_step_end=r["seg_end"], first_pass_minus_annotated_end=r["pass_minus_seg_end"],
                          env_success_first_frame=r["env_success_first_frame"], false_at_seg_start=r["false_at_seg_start"],
                          failure=r.get("failure"), instr=r["instr"], length=r["length"]))
    out[t] = dict(n_demos=a["n"],
                  name_merged_sequences=a["sequences"],
                  step_sequences=a["unit_sequences"],
                  n_distinct_step_sequences=len(a["unit_sequences"]),
                  majority_share=a["majority_share"],
                  n_distinct_stage_sequences=a["n_stage_sequences"], top_stage_sequences=a["stage_sequences"],
                  nonmajority_demos=a["nonmajority_demos"], sentences=a["sentences"],
                  chosen_steps=skills,
                  validation=dict(demos_passed=sum(r["all_passed"] for r in main), demos_total=len(main),
                                  per_step=per_step,
                                  failures=[dict(ep=r["ep"], reason=r.get("failure"), env_success_ever=r["env_success_first_frame"] is not None)
                                            for r in V if not r["all_passed"]],
                                  extra_nonmajority=[dict(ep=r["ep"], all_passed=r["all_passed"]) for r in V if r["extra"]]),
                  demos=demos)
json.dump(out, open(os.path.join(HERE, "report.json"), "w"), indent=1)
for t, o in out.items():
    v = o["validation"]
    print(t, o["chosen_steps"], o["n_distinct_step_sequences"], o["majority_share"], f'{v["demos_passed"]}/{v["demos_total"]}',
          [(s["passed"], s["first_pass_minus_annotated_end"]["min"], s["first_pass_minus_annotated_end"]["max"]) for s in v["per_step"]],
          v["failures"], v["extra_nonmajority"])
