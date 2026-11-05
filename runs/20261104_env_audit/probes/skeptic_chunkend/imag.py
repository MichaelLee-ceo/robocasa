import sys, json, glob
NST = dict(KettleBoiling=2, SteamInMicrowave=4, PackIdenticalLunches=4)
for d in ["20261026_imagine/full_v2", "20261026_imagine/sim_pil"]:
    rs = [json.loads(l) for p in glob.glob(f"/home/michaellee/mclee/robocasa/runs/{d}/results_w*.jsonl") for l in open(p)]
    nc = nmid = nfin_last = 0; committed_mid = []; succ_chunkend = 0; succ = 0
    for r in rs:
        if r.get("condition") != "imagine": continue
        n = NST[r["task"]]
        for e in r["expansions"]:
            if e["k"] != n - 1: continue
            for c in e["cands"]:
                nc += 1
                if c["success"]: nfin_last += 1
                if c["success"] and c["reached"] == e["k"]: nmid += 1
        if r["success"]:
            succ += 1
            ei, ci = r["path"][-1]
            e = r["expansions"][ei]; c = e["cands"][ci]
            if c["success"] and c["reached"] == e["k"] and e["k"] == n - 1:
                committed_mid.append(r["tag"])
            else:
                succ_chunkend += 1
    print(d, "last-stage cands", nc, "with success", nfin_last, "mid-chunk-only", nmid, "| searches success", succ, "committed mid-only", committed_mid)
