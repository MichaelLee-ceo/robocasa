import sys, json, collections
sys.path.insert(0, '.')
from demo_utils import load_demo, n_episodes

def gt_instances(segs):
    out, open_pp = [], {}
    for s in segs:
        st, sk = s["stage"], s["skill"]
        if st == "done":
            continue
        if st == "pick":
            out.append(sk); open_pp[sk] = len(out) - 1
        elif st == "place":
            if sk not in open_pp:
                out.append(sk)
            open_pp.pop(sk, None)
        else:
            if out and out[-1] == sk and st not in ("navigate",):
                continue  # multi-stage execute of the same skill (e.g. press + wait)
            out.append(sk)
    return out

def edit(a, b):
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (a[i-1] != b[j-1]))
    return d[-1][-1]

def main():
    pred = json.load(open(sys.argv[1])); contaminated = set(pred["_meta"].get("contaminated", []))
    rows = []
    for t, p in pred.items():
        if t.startswith("_"):
            continue
        seqs = collections.Counter()
        for ep in range(0, n_episodes(t), 25):
            seqs[tuple(gt_instances(load_demo(t, ep)["segs"]))] += 1
        variants = set(seqs)
        p = [x.rstrip("*") for x in p]
        best = min(variants, key=lambda g: edit(list(g), p))
        e = edit(list(best), p)
        exact = tuple(p) in variants
        skills_ok = set(x for x in p if not x.endswith("*")) <= set(sum(map(list, variants), []))
        rows.append(dict(task=t, contaminated=t in contaminated, pred=p, gt_most_common=list(seqs.most_common(1)[0][0]),
                         n_gt_variants=len(variants), exact=exact, edit=e, norm_edit=round(e / max(len(best), len(p)), 2),
                         pred_uses_nonatomic=any(x.endswith("*") for x in p)))
    for r in rows:
        print(f"{'(C) ' if r['contaminated'] else ''}{r['task']:22s} exact={r['exact']!s:5s} ned={r['norm_edit']:.2f}  GT={' > '.join(r['gt_most_common'])}  ||  PRED={' > '.join(r['pred'])}")
    clean = [r for r in rows if not r["contaminated"]]
    print(f"\nclean tasks: {len(clean)}  exact match {sum(r['exact'] for r in clean)}/{len(clean)}  mean normalized edit distance {sum(r['norm_edit'] for r in clean)/len(clean):.2f}")
    json.dump(rows, open(sys.argv[2], "w"), indent=1)

if __name__ == "__main__":
    main()
