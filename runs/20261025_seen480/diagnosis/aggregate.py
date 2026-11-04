"""Aggregate the per-episode diagnoses (diagnosis/<Task>/<tag>.json) into failure modes.
Primary categories come from the agents (AGENT_BRIEF.md); "other: ..." texts are mapped by keyword. Modes group the
categories by what a fix would have to address. Writes diagnosis/_aggregate.json and prints tables."""
import glob, json, os, re, collections as C

HERE = os.path.dirname(os.path.abspath(__file__))
OTHER = [  # (regex on the "other: ..." text, category)
    (r"spatula slipped|tipped in the grip|shrimp stuck", "slipped_in_hand"),
    (r"missed button press|pushed the open door from its inner", "missed_actuation"),
    (r"no retreat after", "no_release_or_withdraw"),
    (r"wrong destination|navigation stopped", "navigation_error"),
    (r"froze in the tucked", "freeze"),
    (r"knocked the large bowl|dislodged on retreat|caught the pot", "disturbed_scene_object"),
    (r"stir misdirected|blade pointing up|held_beside_stream", "tool_or_process_misaimed"),
    (r"lever stayed jammed", "sim_artifact"),
]
MODES = {
    "grasp_acquisition": ["missed_grasp", "missed_grasp_continued_empty", "stall_pre_grasp"],
    "grasp_retention": ["dropped_in_transit", "slipped_in_hand"],
    "placement": ["placed_off_target", "placed_then_fell_or_slid", "release_timing"],
    "stage_not_terminated": ["stall_holding_at_target", "no_release_or_withdraw"],
    "fixture_actuation": ["actuation_insufficient", "stall_at_fixture", "stall_approach_retreat", "missed_actuation",
                          "wrong_fixture_element"],
    "process_skill": ["tool_or_process_misaimed"],  # stir / scrub / rinse-under-stream done wrong or too little
    "task_level": ["skipped_or_wrong_stage", "wander_away", "navigation_error", "freeze", "wrong_object"],
    "disturbance": ["undid_previous_stage", "disturbed_placed_object", "disturbed_scene_object"],
    "criterion_or_sim": ["official_criterion_issue", "sim_artifact"],
}
CAT2MODE = {c: m for m, cs in MODES.items() for c in cs}


def category(r, task):
    c = (r.get("category") or "").strip()
    if task == "ScrubCuttingBoard" and c == "actuation_insufficient":  # scrubbed too little: a process, not a fixture
        return "tool_or_process_misaimed"
    if c.startswith("other"):
        for pat, cat in OTHER:
            if re.search(pat, c):
                return cat
        return "other"
    return c


rows = []
for p in sorted(glob.glob(f"{HERE}/*/*_single.json")):
    r = json.load(open(p))
    task = os.path.basename(os.path.dirname(p))
    if r["success"]:
        rows.append(dict(task=task, tag=r["tag"], success=True)); continue
    cat = category(r, task)
    rows.append(dict(task=task, tag=r["tag"], success=False, category=cat, mode=CAT2MODE.get(cat, "unmapped"),
                     root_cause_stage=r.get("root_cause_stage"), failed_stage=r.get("failed_stage"),
                     secondary=r.get("secondary") or [], confidence=r.get("confidence")))
json.dump(rows, open(f"{HERE}/_aggregate.json", "w"), indent=1)

fails = [r for r in rows if not r["success"]]
print(f"{len(rows)} episodes, {len(rows) - len(fails)} successes, {len(fails)} failures")
print("\nfailure modes (all tasks):")
mc = C.Counter(r["mode"] for r in fails)
for m, n in mc.most_common():
    cats = C.Counter(r["category"] for r in fails if r["mode"] == m)
    print(f"  {n:3d} {100 * n / len(fails):4.0f}%  {m:22s} " + ", ".join(f"{c} {k}" for c, k in cats.most_common()))
print("\ntask x mode (failures):")
modes = [m for m, _ in mc.most_common()]
ab = {"grasp_acquisition": "acq", "grasp_retention": "ret", "placement": "plc", "stage_not_terminated": "nterm",
      "fixture_actuation": "fix", "process_skill": "proc", "task_level": "task", "disturbance": "dist",
      "criterion_or_sim": "crit"}
print(f"  {'task':22s} {'fail':>4s} " + " ".join(f"{ab.get(m, m)[:5]:>5s}" for m in modes) + "  root-cause stages")
for t in sorted({r["task"] for r in rows}):
    f = [r for r in fails if r["task"] == t]
    c = C.Counter(r["mode"] for r in f)
    st = C.Counter(r["root_cause_stage"] for r in f)
    print(f"  {t:22s} {len(f):4d} " + " ".join(f"{c.get(m, 0) or '.':>5}" for m in modes) + "  "
          + " ".join(f"s{k}:{v}" for k, v in sorted(st.items(), key=lambda x: (x[0] is None, x[0]))))
sec = C.Counter(s.split(":")[0].strip() for r in fails for s in r["secondary"])
print("\nsecondary tags:", ", ".join(f"{k} {v}" for k, v in sec.most_common(10)))
print("confidence:", dict(C.Counter(r["confidence"] for r in fails)))
