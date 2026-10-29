"""Task 1: exact replay. Rebuild the decision point as collect.py does, run rollout() with the saved chunks, and compare
every state with the saved one. Each case is replayed twice (determinism) with the settle version of its row
(v1..v4, see harness.settle), optionally also with another version (`also`).
Also compares the first policy request with the one saved in snap/ (req/state.*, req/video.*).
usage: python -I exact_replay.py cases.json out.jsonl   (cases: [{dir, task, tag, t, c, m, also?}], one task per file)"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402


def main():
    cases = json.load(open(sys.argv[1]))
    out = open(sys.argv[2], "a")
    base, kenv = H.make_env(cases[0]["task"])
    for cs in cases:
        t0 = time.time()
        row = H.find_row(cs["dir"], cs["tag"], cs["t"])
        job = H.find_job(cs["tag"], cs["t"])
        ver = H.VERSION[row["hidden"]]
        saved = np.load(H.roll_path(cs["dir"], cs["tag"], cs["t"], cs["c"], cs["m"]))
        ss = saved["states"]
        res = dict(case=cs, hidden=row["hidden"], version=ver, n_states=len(ss))
        for v in [ver] * 2 + ([cs["also"]] if cs.get("also") else []):
            P = H.Point(base, kenv, cs["task"], cs["tag"], cs["t"], row["k"], job["scene"], version=v, row=row)
            reqs = []
            st, cl = P.run(saved, P.ep[cs["t"]], len(ss) - 1, P.H0, hook=lambda i, req: reqs.append(req) if i == 0 else None)
            n = min(len(st), len(ss))
            d = np.abs(st[:n] - ss[:n]).max(axis=1)
            first = int(np.argmax(d > 0)) if (d > 0).any() else None
            r = dict(version=v, n=n, len_match=len(st) == len(ss), max_diff=float(d.max()), first_nonzero_step=first,
                     diff_at={s: float(d[s]) for s in (0, 1, 16, 32, 64, 128, n - 1) if s < n})
            # first request vs the saved one (only c0 m0 is saved in snap/)
            snap = f"{H.DATA}/{cs['dir']}/snap/{cs['tag']}_t{cs['t']}.npz"
            if os.path.exists(snap) and reqs:
                z = np.load(snap)
                rq = {}
                for k in z.files:
                    if k.startswith("req/") and k[4:] in reqs[0]:
                        a, b = np.asarray(reqs[0][k[4:]]), z[k]
                        if a.dtype.kind in "fiu" and a.shape == b.shape:
                            rq[k[4:]] = float(np.abs(a.astype(float) - b.astype(float)).max())
                r["req0_maxdiff"] = rq
            res.setdefault("runs", []).append(r)
            print(json.dumps(dict(case=f"{cs['tag']} t{cs['t']} c{cs['c']} m{cs['m']}", v=v, max_diff=r["max_diff"],
                                  first=first, req0=r.get("req0_maxdiff"))), flush=True)
        res["wall_s"] = round(time.time() - t0, 1)
        out.write(json.dumps(res) + "\n"); out.flush()


if __name__ == "__main__":
    main()
