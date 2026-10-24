"""Compare per-chunk fingerprints ("digest": [observation sent, action received, sim state after the chunk]) of the same
episodes across determinism-test runs. usage: python compare.py <reference run dir> <run dir> [<run dir> ...]"""
import glob, json, sys


def load(rd):
    out = {}
    for p in glob.glob(f"{rd}/results_w*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            out[r["tag"]] = r
    return out


ref_dir, others = sys.argv[1], sys.argv[2:]
ref = load(ref_dir)
for rd in others:
    cur = load(rd)
    for tag in sorted(ref):
        a, b = ref[tag], cur.get(tag)
        if b is None or "error" in a or "error" in b:
            print(f"{rd} {tag}: missing or error ({(b or {}).get('error')})"); continue
        da, db = a["digest"], b["digest"]
        first = next((i for i, (x, y) in enumerate(zip(da, db)) if x[:3] != y[:3]), None)  # [3] = per-key obs (newer runs only)
        same_len = len(da) == len(db)
        if first is None and same_len:
            verdict = "IDENTICAL"
        elif first is None:
            verdict = f"same prefix, lengths differ ({len(da)} vs {len(db)} chunks)"
        else:
            which = [n for n, x, y in zip(("obs", "action", "state"), da[first], db[first]) if x != y]
            if "obs" in which and len(da[first]) > 3 and len(db[first]) > 3:  # per-key observation fingerprints
                which[0] = "obs[" + ", ".join(k for k in sorted(da[first][3]) if da[first][3][k] != db[first][3].get(k)) + "]"
            verdict = f"DIFFER from chunk {first} (first differing: {', '.join(which)})"
        print(f"{rd} vs {ref_dir} | {tag}: {verdict} | chunks {len(da)}/{len(db)} | success {a['success']}/{b['success']} "
              f"| history {a['history']} / {b['history']} | policy_seed {a.get('policy_seed')}/{b.get('policy_seed')}")
