"""Score eval_tracker.py runs against the oracle (2026-11-06).

usage: python3 coordinator/scripts/score_tracker.py DIR [DIR ...]     (DIR = one model x arm; prints a table, writes
       DIR/score.json)

Progress tracking (every VLM check; the oracle = the record's stages.track at the same chunk end)
  flag acc        the VLM's raw stages_done flags vs the oracle's done flags (every sub-goal of every check)
  false done      checks where the VLM's raw answer calls a sub-goal done that the oracle does not (per check)
  ahead           checks at which the memory's current sub-goal is past the oracle's (it moved on too early): the
                  dangerous error, the coordinator would never retry that sub-goal
  runs ahead      runs in which the memory got ahead of the oracle at least once
  lag             chunks from the oracle completing a sub-goal (first time) to the memory moving past it; "never" when
                  the memory did not move past it by the end
Failure detection (phase 0 failure points: runs/20261105_recovery/phase0/points.json; the oracle supervisor steps in
at the point's chunk, its stall window after the stage began)
  flagged         the VLM reported an event (any event other than "none") on the failed sub-goal (oracle current) at
                  or before the oracle's point
  lead            chunks the first such flag came before the oracle's point (positive = earlier)
  false alarm     checks with an event while the oracle's current sub-goal gets done within the next 16 chunks
                  (progressing), and checks with an event in runs that succeed
  trigger@n       a trigger = n checks in a row reporting an event on the same (oracle) sub-goal, at least 4 chunks
                  after it began (detection/vlm_event.py):
                  failure points caught before the oracle's point and the median lead; false triggers (the sub-goal
                  was done within 16 chunks after) per run"""
import glob, json, os, sys, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTS = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "runs", "20261105_recovery", "phase0", "points.json")))}
PROGRESS_H = 16
MIN_CHUNKS = 4


def oracle_first_done(checks, k):
    c = next((c for c in checks if c["oracle_done"][k]), None)
    return None if c is None else c["chunk"]


def score(d):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json"))) if not f.endswith("score.json")]
    S = dict(dir=d, runs=len(runs), checks=0, errors=0, flags=0, flags_ok=0, false_done=0, ahead=0, runs_ahead=0,
             lag=[], never=0, done_events=0, fa_prog=0, n_prog=0, fa_succ=0, n_succ=0, det=[], seconds=[],
             replay_bad=0, events={})
    for r in runs:
        ch = r["checks"]
        # runs scored before 2026-11-06 16:35 compared lengths, wrong for successes (the coordinator stops at success)
        S["replay_bad"] += not r["replay_matches_record"] and not (r["record_success"] and "success_step" not in r)
        any_ahead = False
        for i, c in enumerate(ch):
            S["checks"] += 1
            if c["seconds"]:
                S["seconds"].append(c["seconds"])
            a = c["answer"] if isinstance(c["answer"], dict) else None
            if c["done"] is None or a is None:
                S["errors"] += 1
                continue
            raw = [bool(x) for x in a.get("stages_done", [])]
            od = c["oracle_done"]
            if len(raw) == len(od):
                S["flags"] += len(od)
                S["flags_ok"] += sum(x == y for x, y in zip(raw, od))
                S["false_done"] += any(x and not y for x, y in zip(raw, od))
            if c["memory_before"] > c["oracle_stage"]:
                S["ahead"] += 1
                any_ahead = True
            ev = a.get("event", "none")
            S["events"][ev] = S["events"].get(ev, 0) + 1
            if r["record_success"]:
                S["n_succ"] += 1
                S["fa_succ"] += ev != "none"
            k = c["oracle_stage"]
            if k < len(od):
                later = [x for x in ch if c["chunk"] < x["chunk"] <= c["chunk"] + PROGRESS_H]
                if any(x["oracle_done"][k] for x in later):
                    S["n_prog"] += 1
                    S["fa_prog"] += ev != "none"
        S["runs_ahead"] += any_ahead
        K = len(ch[0]["oracle_done"]) if ch else 0
        for k in range(K):
            co = oracle_first_done(ch, k)
            if co is None:
                continue
            cm = next((c["chunk"] for c in ch if c["memory_before"] > k or (c["current"] is not None and c["current"] > k
                                                                             and c["done"] is not None)), None)
            if cm is None:
                S["never"] += 1
            else:
                S["lag"].append(cm - co)
        p = POINTS.get(r["run"])
        if p is not None and p["kind"] != "late":
            s, cp = p["stage"], p["chunk"]
            flags = [c for c in ch if c["oracle_stage"] == s and isinstance(c["answer"], dict)
                     and c["answer"].get("event", "none") != "none"]
            first = flags[0]["chunk"] if flags else None
            S["det"].append(dict(run=r["run"], stage=s, point=cp, first_flag=first,
                                 lead=None if first is None else cp - first,
                                 event=flags[0]["answer"].get("event") if flags else None))
    trig = {}
    for nn in (1, 2, 3):
        caught, leads, false_t = 0, [], 0
        for r in runs:
            ch, streak, stage = r["checks"], 0, None
            p = POINTS.get(r["run"])
            first, began = None, {}
            for i, c in enumerate(ch):
                a = c["answer"] if isinstance(c["answer"], dict) else {}
                ev = a.get("event", "none") if c["done"] is not None else "none"
                began.setdefault(c["oracle_stage"], c["chunk"])
                if c["chunk"] - began[c["oracle_stage"]] < MIN_CHUNKS:   # as detection/vlm_event.py: not right after
                    ev = "none"                                            # the sub-goal began
                streak = streak + 1 if (ev != "none" and c["oracle_stage"] == stage) else (1 if ev != "none" else 0)
                stage = c["oracle_stage"]
                if streak >= nn:
                    streak = 0
                    k = c["oracle_stage"]
                    later = [x for x in ch if c["chunk"] < x["chunk"] <= c["chunk"] + PROGRESS_H]
                    if k < len(c["oracle_done"]) and any(x["oracle_done"][k] for x in later):
                        false_t += 1
                    if p is not None and p["kind"] != "late" and k == p["stage"] and first is None:
                        first = c["chunk"]
            if p is not None and p["kind"] != "late" and first is not None and first <= p["chunk"]:
                caught += 1
                leads.append(p["chunk"] - first)
        trig[nn] = dict(caught=caught, lead_median=st.median(leads) if leads else None,
                        false_per_run=round(false_t / max(1, len(runs)), 2))
    n = max(1, S["checks"] - S["errors"])
    det = S["det"]
    flagged = [x for x in det if x["first_flag"] is not None and x["lead"] >= 0]
    lag = S["lag"]
    out = dict(dir=os.path.basename(os.path.dirname(d.rstrip("/"))) + "/" + os.path.basename(d.rstrip("/")),
               runs=S["runs"], checks=S["checks"], errors=S["errors"], replay_bad=S["replay_bad"],
               flag_acc=round(S["flags_ok"] / max(1, S["flags"]), 3), false_done=round(S["false_done"] / n, 3),
               ahead=round(S["ahead"] / n, 3), runs_ahead=S["runs_ahead"],
               lag_median=st.median(lag) if lag else None, lag_p90=sorted(lag)[int(.9 * (len(lag) - 1))] if lag else None,
               lag_negative=sum(x < 0 for x in lag), never=S["never"],
               detected=f"{len(flagged)}/{len(det)}", lead_median=st.median([x["lead"] for x in flagged]) if flagged else None,
               fa_progressing=round(S["fa_prog"] / max(1, S["n_prog"]), 3), fa_success_runs=round(S["fa_succ"] / max(1, S["n_succ"]), 3),
               trigger=trig, events=S["events"], latency_median=st.median(S["seconds"]) if S["seconds"] else None, detections=det)
    json.dump(out, open(os.path.join(d, "score.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    cols = ["dir", "runs", "checks", "errors", "flag_acc", "false_done", "ahead", "runs_ahead", "lag_median", "lag_p90",
            "never", "detected", "lead_median", "fa_progressing", "fa_success_runs", "latency_median"]
    print(" | ".join(cols))
    for d in sys.argv[1:]:
        o = score(d)
        print(" | ".join(str(o[c]) for c in cols))
        print("    triggers:", o["trigger"], " events:", o["events"])
