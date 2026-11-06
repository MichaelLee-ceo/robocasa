"""Score the VLM pilot: answers (pilot_vlm/packets/<qid>/answer.json) against the simulator's ground truth
(pilot_vlm/truth.jsonl). Prints the per-query table and the summary; writes pilot_vlm/scored.jsonl.
usage: python3 score_vlm_pilot.py"""
import collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "pilot_vlm")


def load_answer(qid):
    p = os.path.join(D, "packets", qid, "answer.json")
    if not os.path.exists(p):
        return {"error": "no answer"}
    txt = open(p).read()
    s, e = txt.find("{"), txt.rfind("}")
    try:
        return json.loads(txt[s:e + 1])
    except Exception as x:
        return {"error": str(x)}


def main():
    T = [json.loads(l) for l in open(os.path.join(D, "truth.jsonl"))]
    rows, st = [], collections.Counter()
    for t in sorted(T, key=lambda r: (r["task"], r["seed"], r["chunk"])):
        a = load_answer(t["qid"])
        r = dict(t, answer=a)
        if "error" in a:
            st["unanswered"] += 1
            rows.append(r)
            continue
        pred = [bool(x) for x in a.get("stages_done", [])]
        truth = t["stages_done"]
        if len(pred) != len(truth):
            st["wrong_length"] += 1
        for p, g in zip(pred, truth):
            st[f"stage_{'tp' if p and g else 'fp' if p else 'fn' if g else 'tn'}"] += 1
        oracle = sorted({e["type"] for e in t["events"]})
        r.update(pred=pred, oracle_events=oracle, vlm_event=a.get("event"))
        st["oracle_event_windows"] += bool(oracle)
        st["vlm_flags_event"] += a.get("event") != "none"
        st["both_event"] += bool(oracle) and a.get("event") != "none"
        st["oracle_event_vlm_same_type"] += a.get("event") in oracle
        rows.append(r)
    with open(os.path.join(D, "scored.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{'task':20s} {'s':>2s} {'chunk':>5s} {'truth stages':14s} {'vlm stages':14s} {'oracle events':28s} vlm event (conf)")
    for r in rows:
        a = r["answer"]
        if "error" in a:
            print(f"{r['task']:20s} {r['seed']:>2d} {r['chunk']:>5d} ERROR {a['error']}")
            continue
        fmt = lambda v: "".join("x" if b else "." for b in v)
        print(f"{r['task']:20s} {r['seed']:>2d} {r['chunk']:>5d} {fmt(r['stages_done']):14s} {fmt(r['pred']):14s} "
              f"{','.join(r['oracle_events']) or '-':28s} {r['vlm_event']} ({a.get('confidence')})  raw={r['raw']}")
    tp, fp, fn, tn = (st[f"stage_{k}"] for k in ("tp", "fp", "fn", "tn"))
    n = tp + fp + fn + tn
    print(f"\nstage flags: {n} judged; accuracy {(tp + tn) / max(n, 1):.2f}; done recall {tp / max(tp + fn, 1):.2f}; "
          f"not-done recall {tn / max(tn + fp, 1):.2f}; balanced {(tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2:.2f}; "
          f"false 'done' {fp}, missed 'done' {fn}")
    print(f"events: oracle-event windows {st['oracle_event_windows']}; VLM flagged an event in {st['vlm_flags_event']} "
          f"of {len(rows) - st['unanswered']}; both {st['both_event']}; same type as an oracle event "
          f"{st['oracle_event_vlm_same_type']}; unanswered {st['unanswered']}")


if __name__ == "__main__":
    main()
