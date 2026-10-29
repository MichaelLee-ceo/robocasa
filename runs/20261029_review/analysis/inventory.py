"""Inventory of every results_w*.jsonl row under runs/20261026_critic_data: per dir, the hidden tags, N/M/L, error rows,
duplicate error-free rows of one (tag, t), k vs k_replay, partial lines. Read-only.
usage: python -I inventory.py"""
import collections, glob, json, os

DATA = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"


def rows_of(d):
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, d, "results_w*.jsonl"))):
        for n, line in enumerate(open(p)):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                out.append(dict(_bad=True, _file=os.path.basename(p), _line=n))
                continue
            r["_file"] = os.path.relpath(p, DATA); r["_line"] = n
            out.append(r)
    return out


def main():
    dirs = sorted({os.path.relpath(os.path.dirname(p), DATA) for p in glob.glob(os.path.join(DATA, "**", "results_w*.jsonl"), recursive=True)})
    for d in dirs:
        rs = rows_of(d)
        bad = [r for r in rs if r.get("_bad")]
        err = [r for r in rs if not r.get("_bad") and "error" in r]
        ok = [r for r in rs if not r.get("_bad") and "error" not in r]
        hid = collections.Counter(r.get("hidden") for r in ok)
        nm = collections.Counter((r.get("N"), r.get("M")) for r in ok)
        kk = [(r["tag"], r["t"], r["k"], r.get("k_replay")) for r in ok if r.get("k_replay") is not None and r["k_replay"] != r["k"]]
        dup = collections.Counter((r["tag"], r["t"]) for r in ok)
        dups = {k: v for k, v in dup.items() if v > 1}
        err_keys = {(r["tag"], r["t"]) for r in err}
        err_unresolved = sorted(k for k in err_keys if k not in dup)
        print(f"{d:40s} ok {len(ok):4d} err {len(err):3d} (unresolved {len(err_unresolved)}) bad-lines {len(bad)} hidden {dict(hid)} N,M {dict(nm)}")
        if kk:
            print(f"    k != k_replay at {len(kk)} rows, e.g. {kk[:6]}")
        if dups:
            print(f"    duplicate error-free rows for {len(dups)} (tag, t), e.g. {list(dups.items())[:6]}")
            for key in list(dups)[:4]:
                same = [r for r in ok if (r["tag"], r["t"]) == key]
                print("      ", key, [(r["_file"], r["_line"], r.get("hidden"), round(sum(x["done"] for c in r["cands"] for x in c["rollouts"]) / sum(len(c["rollouts"]) for c in r["cands"]), 3)) for r in same])
        if err_unresolved:
            print(f"    error rows never redone: {err_unresolved[:10]}")


if __name__ == "__main__":
    main()
