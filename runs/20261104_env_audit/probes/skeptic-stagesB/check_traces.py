import json, gzip, sys, numpy as np
A = "/home/michaellee/mclee/robocasa/runs/20261025_seen480/analysis"
rows = [json.loads(l) for l in open(f"{A}/stages.jsonl")]
rows = {r["tag"]: r for r in rows if "error" not in r}
tasks = sys.argv[1].split(",")
for task in tasks:
    n_compl = 0; n_prev_false = 0; det = []
    mism = []
    for tag, r in sorted(rows.items()):
        if r["task"] != task: continue
        with gzip.open(f"{A}/{task}/{tag}_trace.json.gz", "rt") as f:
            rec = json.load(f)
        st = np.array(rec["stage"])
        T = len(st)
        # prev_ok at each tracker completion of stage k>=1, k < last
        nst = st.shape[1]
        for t, k in r["history"]:
            if 1 <= k < nst - 1:
                n_compl += 1
                ok = all(st[t][q] for q in range(k))
                if not ok:
                    n_prev_false += 1
                    extra = {}
                    if task == "SearingMeat":
                        extra = dict(pan_eef=rec["obj"]["pan"]["eef_dist"][t], pan_at=rec["at"][0][t]["pan"], pan_far=rec["obj"]["pan"]["far"][t])
                    det.append((tag, t, k, [bool(st[t][q]) for q in range(k)], extra))
        reached = len(r["history"])
        fis = r["first_incomplete_stage"]
        if fis is not None and fis < reached:
            v = r["stages"][fis].get("details")
            mism.append((tag, fis, reached, r["success"], [d.get("verdict", d.get("kind")) for d in (v or [])]))
    print(task, "completions(1<=k<last):", n_compl, "prev_ok False:", n_prev_false)
    for d in det: print("  ", d)
    print(" first_incomplete < reached:")
    for m in mism: print("  ", m)
