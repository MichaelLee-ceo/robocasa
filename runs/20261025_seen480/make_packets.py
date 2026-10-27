"""Per-episode data packets for the video-based diagnosis agents.

Reads the rerun results (run/results_w*.jsonl), the program analysis (analysis/stages.jsonl) and the per-frame traces
(analysis/<task>/<tag>_trace.json.gz) and writes packets/<task>/<tag>.json (compact, for reading) plus
packets/<task>/_task.json (stage list and typical stage durations from this task's successful episodes).
Units: 1 step = 1 control step = 0.05 s; a policy chunk = 16 steps; the video keeps one frame every 4 steps.
usage: python make_packets.py [comma-separated tasks]"""
import glob, gzip, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from stage_trace import STAGE_SPECS, _holds  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "prev", "scripts"))
from subgoal_defs import DEFS  # noqa: E402  (plain-language stage definitions, written from the same simulator checks)


def intervals(mask):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append([s, i - 1]); s = None
    return out


def close_events(width, touch_by_obj, pos_by_obj):
    """gripper closing events from the finger width (sum of |finger joint qpos|, 0.08 open): a drop of >= 0.02 within 10
    steps; reports the width 15 steps later, whether it touched none of the task objects (on_nothing), which ones it touched, and whether that
    object rose >= 1 cm in the next 40 steps."""
    ev, T, c = [], len(width), 0
    while c < T - 10:
        if width[c] - width[c + 10] >= 0.02:
            after = min(c + 15, T - 1)
            touching = [o for o, t in touch_by_obj.items() if any(t[c:after + 1])]
            lifted = [o for o in touching if max(p[2] for p in pos_by_obj[o][after:min(after + 40, T)]) - pos_by_obj[o][after][2] >= 0.01]
            ev.append(dict(step=c, width_after=round(width[after], 3), on_nothing=not touching,
                           touching=touching, lifted=lifted))
            c += 20
        else:
            c += 1
    return ev


results = {json.loads(l)["tag"]: json.loads(l) for p in glob.glob(f"{HERE}/run/results_w*.jsonl") for l in open(p)}
ADIR = os.environ.get("ANALYSIS", f"{HERE}/analysis")  # program analysis dir (stages.jsonl + traces)
ODIR = os.environ.get("PACKETS", f"{HERE}/packets")
analysis = {r["tag"]: r for r in map(json.loads, open(f"{ADIR}/stages.jsonl")) if "error" not in r}
tasks = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted({r["task"] for r in analysis.values()})
for task in tasks:
    eps = sorted((t for t, r in analysis.items() if r["task"] == task), key=lambda t: int(t.split("_s")[1].split("_")[0]))
    specs = STAGE_SPECS[task]
    # typical per-stage durations (steps from the previous stage's first completion) in successful episodes
    dur = defaultdict(list)
    for t in eps:
        a = analysis[t]
        if not a["success"]:
            continue
        prev = 0
        for s in a["stages"]:
            if s["first_true"] is not None:
                dur[s["stage"]].append(s["first_true"] - prev); prev = max(prev, s["first_true"])
    typical = {k: dict(median=int(np.median(v)), p90=int(np.percentile(v, 90)), n=len(v)) for k, v in dur.items()}
    os.makedirs(f"{ODIR}/{task}", exist_ok=True)
    r0 = results[eps[0]]
    json.dump(dict(task=task, instruction=r0["task_instr"], stages=[f"{k}: {s}" for k, s in enumerate(r0["skills"])],
                   stage_definitions=[dict(stage=k, counts_as_complete=d[0], does_not_count=d[1]) for k, d in enumerate(DEFS.get(task, []))],
                   horizon=r0["horizon"], episodes=len(eps), successes=sum(analysis[t]["success"] for t in eps),
                   typical_stage_steps_in_successes=typical),
              open(f"{ODIR}/{task}/_task.json", "w"), indent=1)
    for t in eps:
        a, r = analysis[t], results[t]
        rec = json.load(gzip.open(f"{ADIR}/{task}/{t}_trace.json.gz"))
        st = np.array(rec["stage"])
        T = len(st)
        stages, prev = [], 0
        for s in a["stages"]:
            k = s["stage"]
            first = s["first_true"]
            ivs = intervals(st[:, k])
            undone = []
            for _, e_ in ivs:
                if e_ < T - 1:
                    why = "condition dropped"
                    if specs[k]["kind"] == "pnp":
                        objs_k = specs[k]["objs"]
                        still = [o for o in objs_k if rec["at"][k][e_ + 1][o]]
                        why = ("objects still at target - only the release part broke (gripper back within 25 cm or touching)"
                               if still and len(still) == sum(rec["at"][k][e_][o] for o in objs_k) else "an object left the target")
                    undone.append(dict(step=e_ + 1, reason=why))
            stages.append(dict(stage=k, skill=s["skill"], kind=s["kind"], first_complete_step=first,
                               complete_at_end=s["true_at_end"], complete_intervals=ivs, undone_events=undone,
                               steps_since_previous_stage=(first - prev) if first is not None else None,
                               window_start=s["window_start"], typical_in_successes=typical.get(k)))
            if first is not None:
                prev = max(prev, first)
        objects = {}
        for k, sp in enumerate(specs):
            if sp["kind"] != "pnp":
                continue
            for o in sp["objs"]:
                if o in objects:
                    continue
                ro = rec["obj"][o]
                holds, brief = [], 0
                for g, e in _holds(ro, 0, T):
                    lift = max(p[2] for p in ro["pos"][g:e]) - ro["pos"][g][2]
                    if lift < 0.01 and sum(ro["grasp"][g:e]) < 3:  # 1-2 step contacts / pushes without lift
                        brief += 1
                        continue
                    rel = min(e, T - 1); settle = min(rel + 40, T - 1)
                    holds.append(dict(grasp_step=g, release_step=e if e < T else None,
                                      lift_m=round(max(p[2] for p in ro["pos"][g:e]) - ro["pos"][g][2], 3),
                                      at_target_at_release=rec["at"][k][rel][o], dist_to_target_at_release_m=rec["dist"][k][rel][o],
                                      height_drop_after_release_m=round(ro["pos"][rel][2] - ro["pos"][settle][2], 3),
                                      at_target_2s_after=rec["at"][k][settle][o], dist_2s_after_m=rec["dist"][k][settle][o]))
                objects[o] = dict(target_of_stage=k, start_pos=ro["pos"][0], end_pos=ro["pos"][-1],
                                  at_target_intervals=intervals([rec["at"][k][f][o] for f in range(T)]),
                                  dist_to_target_start_m=rec["dist"][k][0][o], dist_to_target_end_m=rec["dist"][k][T - 1][o],
                                  min_eef_dist_to_centre_m=min(ro["eef_dist"]), gripper_touch_intervals=intervals(ro["touch"]),
                                  holds=holds, brief_contacts_without_lift=brief,
                                  tilt_deg_start=ro["tilt"][0], tilt_deg_end=ro["tilt"][-1], tilt_deg_max=max(ro["tilt"]),
                                  tipped_over_intervals=intervals([t_ > 45 for t_ in ro["tilt"]]))
        fixtures = {}
        for k, sp in enumerate(specs):
            if sp["kind"] != "fixture":
                continue
            v = rec["fx_value"][k]
            fixtures[f"stage{k}:{sp['fixture']}"] = dict(
                value_every_16_steps=v[::16] + ([v[-1]] if (T - 1) % 16 else []),
                done_intervals=intervals(rec["fx_done"][k]), gripper_touch_intervals=intervals(rec["fx_touch"][sp["fixture"]]),
                state_every_32_steps=rec["fx_state"][k][::32] + ([rec["fx_state"][k][-1]] if (T - 1) % 32 else []),
                state_end=rec["fx_state"][k][-1])
        process = {f"stage{k}": rec["progress"][k][-1] for k, sp in enumerate(specs) if sp["kind"] == "process"}
        last = a["stages"][-1]["first_true"]
        gripper = dict(width_every_8_steps=rec["grip_width"][::8],
                       close_events=close_events(rec["grip_width"], {o: v["touch"] for o, v in rec["obj"].items()},
                                                 {o: v["pos"] for o, v in rec["obj"].items()}))
        fp = r.get("scene_fp", {})
        # rollout's own stage log (sequential tracker, evaluated live at chunk boundaries) vs the replay
        consistent = [dict(stage=k_, logged_step=t_, replay_holds_there=bool(st[min(t_, T - 1), k_])) for t_, k_ in r["history"]]
        packet = dict(tag=t, task=task, instruction=r["task_instr"], success=r["success"], steps=r["steps"], horizon=r["horizon"],
                      success_step=(last if last is not None else r["steps"]) if r["success"] else None,
                      online_stage_log=r["history"], replay_matches_online_log=all(c_["replay_holds_there"] for c_ in consistent),
                      online_vs_replay=consistent,
                      scene=dict(layout=fp.get("layout"), style=fp.get("style"), objects=fp.get("objects")),
                      gripper=gripper,
                      units="1 step = 0.05 s (20 Hz); policy chunk = 16 steps; video has one frame every 4 steps",
                      first_incomplete_stage=a["first_incomplete_stage"], stages=stages, objects=objects, fixtures=fixtures,
                      process_progress_end=process,
                      official_labeler=dict(note="official subtask labeler, built for successful demos; on failures its "
                                                 "boundaries can be wrong", segments=a["labeler_segments"],
                                            issues=a["labeler_issues"]),
                      video=f"run/{task}/{t}.mp4")
        json.dump(packet, open(f"{ODIR}/{task}/{t}.json", "w"), indent=1, default=str)
    print(task, len(eps), "packets")
