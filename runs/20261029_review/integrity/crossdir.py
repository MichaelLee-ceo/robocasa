"""Determinism across runs: the same decision point (tag, t) collected in several dirs.
Seeds depend only on (tag, t, candidate) for chunk 0, so with the same observation a deterministic server returns the
same chunk 0, and with the same hidden state the same states[0..16].
1. per candidate (continuation 0), pairs of dirs: obs_fp equal? chunk 0 equal? states[0..16] equal?
2. the saved first request of c0 m0 (snap req/*) per key: bitwise equal? how many pixels differ, by how much?
   and the 480x480 snapshot renders (rgb/*, depth/*, depth_env/*)
Writes crossdir.json. usage: python -I crossdir.py [workers=3]"""
import collections, itertools, json, os, sys, zipfile, io
from multiprocessing import get_context
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, HERE  # noqa: E402
from analyze import load  # noqa: E402


def snapcmp(args):
    tag, t, da, db = args
    out = dict(tag=tag, t=t, a=da, b=db, keys={})
    try:
        za = np.load(f"{DATA}/{da}/snap/{tag}_t{t}.npz", allow_pickle=True)
        zb = np.load(f"{DATA}/{db}/snap/{tag}_t{t}.npz", allow_pickle=True)
        for k in sorted(set(za.files) & set(zb.files)):
            a, b = za[k], zb[k]
            if a.dtype == object or a.dtype.kind in "US":
                out["keys"][k] = "eq" if np.array_equal(a, b) else "neq"
                continue
            if a.shape != b.shape:
                out["keys"][k] = f"shape {a.shape} vs {b.shape}"; continue
            if np.array_equal(a, b):
                out["keys"][k] = "eq"; continue
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            if k.startswith(("req/video", "rgb/")):
                px = d.reshape(-1, d.shape[-1]).max(1) if d.shape[-1] == 3 else d.ravel()
                out["keys"][k] = dict(px=int((px > 0).sum()), of=int(px.size), max=float(d.max()))
            else:
                out["keys"][k] = dict(n=int((d > 0).sum()), of=int(d.size), max=float(d.max()))
        out["only_a"] = sorted(set(za.files) - set(zb.files)); out["only_b"] = sorted(set(zb.files) - set(za.files))
    except Exception as e:
        out["err"] = repr(e)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    recs, _ = load()
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["tag"], r["t"])].append(r)
    pairs = collections.Counter()
    ex = collections.defaultdict(list)
    for key, rs in by.items():
        if len(rs) < 2:
            continue
        for ra, rb in itertools.combinations(rs, 2):
            ca = {o["i"]: o for o in ra["rolls"] if o["m"] == 0 and "h_c0" in o}
            cb = {o["i"]: o for o in rb["rolls"] if o["m"] == 0 and "h_c0" in o}
            same_h = ra["hidden"] == rb["hidden"]
            kind = f"{ra['status']}|{rb['status']}" if ra["status"] <= rb["status"] else f"{rb['status']}|{ra['status']}"
            for i in set(ca) & set(cb):
                a, b = ca[i], cb[i]
                fp = a["obs_fp"] == b["obs_fp"]; c0 = a["h_c0"] == b["h_c0"]; s16 = a["h_s16"] == b["h_s16"]
                pairs[("all", fp, c0)] += 1
                pairs[(kind, fp, c0)] += 1
                if c0:
                    pairs[("c0eq_s16", same_h, a["cm0"] >= 0.5, s16)] += 1
                    if same_h and not s16:
                        ex["same_hidden_same_c0_diff_s16"].append((key, ra["dir"], rb["dir"], i, ra["hidden"]))
                if fp and not c0:
                    ex["same_obs_diff_c0"].append((key, ra["dir"], rb["dir"], i))
    print("candidate pairs across dirs: (kind, obs_fp equal, chunk0 equal) -> count")
    for k, v in sorted(pairs.items(), key=lambda x: str(x[0])):
        print("  ", k, v)
    for k, v in ex.items():
        print(k, len(v), v[:8])
    # snap comparisons: every (tag, t) in >= 2 dirs, one pair per dir pair
    jobs = []
    for key, rs in by.items():
        ds = sorted({r["dir"] for r in rs})
        for da, db in itertools.combinations(ds, 2):
            jobs.append((key[0], key[1], da, db))
    with get_context("spawn").Pool(nw) as pool:
        snaps = list(pool.imap_unordered(snapcmp, jobs, chunksize=4))
    agg = collections.defaultdict(collections.Counter)
    pxs = collections.defaultdict(list)
    for s in snaps:
        if "err" in s:
            agg["_err"]["n"] += 1; continue
        for k, v in s["keys"].items():
            kk = k.split("/")[0] + "/" + (k.split("/")[1] if k.startswith(("req/", "rgb/", "depth/", "depth_env/", "K/", "T/")) else "")
            if v == "eq":
                agg[kk]["eq"] += 1
            else:
                agg[kk]["neq"] += 1
                if isinstance(v, dict):
                    pxs[kk].append(v.get("px", v.get("n")))
                    agg[kk]["max_diff_max"] = max(agg[kk]["max_diff_max"], v["max"])
    print(f"snap pairs compared: {len(snaps)}")
    for k in sorted(agg):
        p = pxs.get(k)
        print(f"  {k:45s} {dict(agg[k])}" + (f" differing px/elements: median {int(np.median(p))}, p95 {int(np.percentile(p, 95))}, max {max(p)}" if p else ""))
    json.dump(dict(pairs={str(k): v for k, v in pairs.items()}, examples=ex, snaps=snaps), open(os.path.join(HERE, "crossdir.json"), "w"))


if __name__ == "__main__":
    main()
