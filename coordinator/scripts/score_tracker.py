"""Score eval_tracker.py runs of the to-do list tracker against the oracle (2026-11-07).

usage: python3 coordinator/scripts/score_tracker.py DIR [DIR ...]   (DIR = one model x window; prints a table, writes
       DIR/score.json). The tracker of 2026-11-06 (all sub-goals at once) was scored by the version at commit d93b364.

The oracle = the record's stages.track at the same chunk end. Items are compared by count (any-order items may be
ticked in any order, and PackIdenticalLunches' oracle stages count placements): with n items ticked, "the next item
is done" = the oracle has completed stage n at least once by this chunk (its first completion), so that a stage
reading undone only because the hand came near does not count against the VLM; the i-th tick is compared with the
oracle's i-th stage. Runs whose plan has another number of items than the oracle's are counted and left out of the
item scores.

Plan          runs whose to-do list has the oracle's number of items
Item check    accuracy of "done" on the current item; false done (says done, the oracle has not
              completed it); missed (the oracle has, it says not)
Ticks         premature: the memory ticked an item before the oracle completed it (the dangerous error: in a closed
              loop that item is never retried); runs with one; lag from the oracle's completion to the tick (chunks);
              never registered: items the oracle completed (at least two checks before the run's end: a run stops at
              the official success) that the memory never got past
Hand-off      every hand-off check against the oracle: passed right / passed wrong (a premature tick) / blocked right
              (a premature tick prevented) / blocked wrong (a delay)
Stall         detection/progress.py ProgressStall(n) replayed on the logged progress dots (n checks in a row not
              advancing, at least 4 chunks into the attempt; again only after n more dots): at the phase 0 failure
              points (runs/20261105_recovery/phase0/points.json, not "late"), caught = a stall on the failed stage
              (memory and oracle both on it) at or before the oracle supervisor's point (its p90 window), lead = how
              many chunks before; false = a stall while the oracle completes the memory's current item within the
              next 16 chunks (it was progressing), per run"""
import glob, json, os, sys, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POINTS = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "runs", "20261105_recovery", "phase0", "points.json")))}
PROGRESS_H = 16
MIN_CHUNKS = 4
STALL_N = (3, 4, 6, 8)


def first_done(checks, K):
    """oracle stage -> chunk of its first completion (None: never)"""
    return [next((c["chunk"] for c in checks if c["oracle_done"][k]), None) for k in range(K)]


def item_answer(c):
    a = next((x["answer"] for x in c["calls"] if x["module"] == "tracker"), None)
    return a if isinstance(a, dict) and "progress" in a else None


def said_done(a):
    """the answer calls an open item done (a bool, or {"<item>": bool})"""
    d = a.get("done", a["progress"] == "done")
    return any(v is True or str(v).strip().lower() == "true" for v in d.values()) if isinstance(d, dict) else bool(d)


def n_done(c):
    return c.get("n_done", c["memory_before"])


def stalls(checks, n):
    """[(chunk, item)] where ProgressStall(n) would fire, from the logged item answers"""
    out, dots, used, start = [], {}, {}, {}
    for c in checks:
        a = item_answer(c)
        key = (c["memory_before"], c["attempt"])
        start.setdefault(key, c["chunk"])
        if a is None:
            continue
        d = dots.setdefault(key, [])
        d.append(a)
        fresh = d[used.get(key, 0):]
        if len(fresh) >= n and c["chunk"] - start[key] >= MIN_CHUNKS and \
                not any(x["advancing"] or said_done(x) or x["progress"] == "done" for x in fresh[-n:]):
            used[key] = len(d)
            out.append((c["chunk"], c["memory_before"]))
    return out


def score(d):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json"))) if not f.endswith("score.json")]
    S = dict(runs=len(runs), plan_ok=0, plan_by_task={}, checks=0, errors=0, n=0, acc=0, false_done=0, missed=0,
             ticks=0, premature=0, runs_premature=0, lag=[], never=0, completions=0, ho=dict(pass_ok=0, pass_wrong=0,
             block_ok=0, block_wrong=0), prev_bad=0, prev_bad_true=0, seconds=[], images=[], replay_bad=0)
    stall = {n: dict(caught=0, leads=[], false=0, points=0) for n in STALL_N}
    for r in runs:
        ch, K = r["checks"], len(r["oracle_plan"])
        plan = r["memory"]["plan"]
        pt = S["plan_by_task"].setdefault(r["task"], [0, 0])
        pt[1] += 1
        S["replay_bad"] += not r["replay_matches_record"]
        for c in ch:
            S["checks"] += 1
            for x in c["calls"]:
                if x.get("seconds"):
                    S["seconds"].append(x["seconds"])
                if x["module"] == "tracker" and x.get("n_images"):
                    S["images"].append(x["n_images"])
            S["errors"] += c["done"] is None
        if len(plan) != K:
            continue
        pt[0] += 1
        S["plan_ok"] += 1
        fd = first_done(ch, K)
        for c in ch:
            a, k = item_answer(c), n_done(c)
            if a is None or k >= K:
                continue
            truth = fd[k] is not None and fd[k] <= c["chunk"]
            pred = said_done(a)
            S["n"] += 1
            S["acc"] += pred == truth
            S["false_done"] += pred and not truth
            S["missed"] += truth and not pred
            if k > 0 and not a["previous_ok"]:
                S["prev_bad"] += 1
                S["prev_bad_true"] += not c["oracle_done"][k - 1]
            for x in c["calls"]:
                if x["module"] == "handoff" and isinstance(x["answer"], dict):
                    ok = bool(x["answer"].get("ready"))
                    S["ho"][("pass_" if ok else "block_") + ("ok" if ok == truth else "wrong")] += 1
        prem = False
        for k, mv in enumerate(sorted(r["memory"]["moved_on"], key=lambda e: e["chunk"])):    # the k-th tick
            if k >= K:
                continue
            S["ticks"] += 1
            if fd[k] is None or fd[k] > mv["chunk"]:
                S["premature"] += 1
                prem = True
            else:
                S["lag"].append(mv["chunk"] - fd[k])
        S["runs_premature"] += prem
        final, last = len(r["memory"]["moved_on"]), ch[-1]["chunk"] if ch else 0
        for k in range(K):
            if fd[k] is not None and fd[k] <= last - 2:      # completions with two checks left to see them
                S["completions"] += 1
                S["never"] += final <= k
        p = POINTS.get(r["run"])
        for n in STALL_N:
            sl = stalls(ch, n)
            by_chunk = {c["chunk"]: c for c in ch}
            for cc, k in sl:
                k = n_done(by_chunk[cc])
                if k < K and fd[k] is not None and cc < fd[k] <= cc + PROGRESS_H:
                    stall[n]["false"] += 1
            if p is not None and p["kind"] != "late":
                stall[n]["points"] += 1
                hit = [cc for cc, k in sl if n_done(by_chunk[cc]) == p["stage"] and by_chunk[cc]["oracle_stage"] == p["stage"]
                       and cc <= p["chunk"]]
                if hit:
                    stall[n]["caught"] += 1
                    stall[n]["leads"].append(p["chunk"] - hit[0])
    n = max(1, S["n"])
    out = dict(dir=os.path.basename(os.path.dirname(d.rstrip("/"))) + "/" + os.path.basename(d.rstrip("/")),
               runs=S["runs"], plan_ok=S["plan_ok"], plan_by_task={t: f"{a}/{b}" for t, (a, b) in S["plan_by_task"].items()},
               checks=S["checks"], errors=S["errors"], replay_bad=S["replay_bad"],
               item_acc=round(S["acc"] / n, 3), false_done=round(S["false_done"] / n, 3), missed=round(S["missed"] / n, 3),
               ticks=S["ticks"], premature=S["premature"], runs_premature=S["runs_premature"],
               lag_median=st.median(S["lag"]) if S["lag"] else None,
               lag_p90=sorted(S["lag"])[int(.9 * (len(S["lag"]) - 1))] if S["lag"] else None,
               never=f"{S['never']}/{S['completions']}", handoff=S["ho"],
               previous_bad=f"{S['prev_bad_true']}/{S['prev_bad']} true",
               stall={n_: dict(caught=f"{v['caught']}/{v['points']}", lead_median=st.median(v["leads"]) if v["leads"] else None,
                               false_per_run=round(v["false"] / max(1, S["plan_ok"]), 2)) for n_, v in stall.items()},
               latency_median=st.median(S["seconds"]) if S["seconds"] else None,
               images_median=st.median(S["images"]) if S["images"] else None)
    json.dump(out, open(os.path.join(d, "score.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    cols = ["dir", "runs", "plan_ok", "checks", "errors", "item_acc", "false_done", "missed", "ticks", "premature",
            "runs_premature", "lag_median", "lag_p90", "never", "latency_median", "images_median"]
    print(" | ".join(cols))
    for d in sys.argv[1:]:
        o = score(d)
        print(" | ".join(str(o[c]) for c in cols))
        print("    plan:", o["plan_by_task"], " hand-off:", o["handoff"], " previous_ok false:", o["previous_bad"])
        print("    stall:", o["stall"])
