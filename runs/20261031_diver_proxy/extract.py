"""DiVeR's candidate features at the critic-data points (step 1 of README.md).

For every point of the chosen data sets (../20261028_critical/data.py), the policy request saved in its snapshot is sent
to ../scripts/serve_features.py once per noise seed:
  k < 8   the seeds of the 8 labelled candidates' first chunks: collect.py seed(tag, t, i, m=0, ci=0) =
          sha256(f"{tag}|critic|t{t}|c{i}|m0|0") (the salt only enters later chunks);
  k >= 8  new seeds sha256(f"{tag}|diver|t{t}|k{k}"), up to DiVeR's K = 32 (no labels; for the dispersion only).
The snapshot holds candidate 0's first request. At many points the other candidates' first requests differ from it in
some camera images (GPU render noise; state and instruction are equal), so a labelled candidate is reproduced bit for bit
only where its request equals candidate 0's (row obs_fp). Every labelled candidate is compared with its recorded first
chunk (roll/<tag>_t<t>_c<i>_m0.npz) and the max |difference| is logged. Where the requests are equal and the row was
collected by the current servers (hidden = settled-v5), the chunk must be identical (else the run stops, unless
--no_strict). Older rows (settled, -basefix, -v3, -v4; 2026-10-27/28) were sampled by an earlier server setup: same
request and seed give chunks within about 0.01 of the recorded ones, identically on GPU0 and GPU3.
Out: <out>/<task>/<ds>/<tag>_t<t>.npz with every "_feat*" array of the server, stacked over seeds and named without the
underscore (feat: K x 16 x 1024 float32; with the clean-pass server also feat_clean K x 16 x 1024 and feat_steps
K x 4 x 1024), seed (K), action.<key> (K x 16 x dim); one log line per point in <out>/log_w<W>.jsonl. Points already
written are skipped. feats/ holds the first extraction (K = 32 at confirm / control); feats_clean/ the clean-pass one
(K = 8 everywhere).
usage: python -I extract.py --port 5791 [--sets confirm,control,dense] [--K 32] [--K_dense 8] [--worker W --n_workers N]
       [--limit N]   (with the groot python; ROBOCASA_ROOT set)
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = (os.environ.get("ROBOCASA_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
sys.path[:0] = [os.path.join(HERE, "..", "20261028_critical"), f"{ROOT}/third_party/Isaac-GR00T"]
import data  # noqa: E402
from gr00t.eval.service import BaseInferenceClient  # noqa: E402

ANN = "annotation.human.task_description"


def fingerprint(d):
    """atomic_eval.fingerprint (that module imports the simulator)"""
    m = hashlib.sha1()
    for k in sorted(d):
        m.update(str(k).encode()); m.update(np.ascontiguousarray(np.asarray(d[k])).tobytes())
    return m.hexdigest()[:12]


def seed_of(tag, t, k):
    key = f"{tag}|critic|t{t}|c{k}|m0|0" if k < 8 else f"{tag}|diver|t{t}|k{k}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def request(r):
    z = np.load(data.snap(r))
    req = {k[4:]: z[k] for k in z.files if k.startswith("req/")}
    req[ANN] = tuple(str(x) for x in req[ANN])  # sent as a tuple (groot_eval.batch); same fingerprint
    if fingerprint(req) != r["cands"][0]["rollouts"][0]["obs_fp"]:
        raise RuntimeError(f"{r['tag']} t{r['t']}: the snapshot's request is not candidate 0's")
    return req


def extract(client, r, K, strict):
    req, tag, t = request(r), r["tag"], r["t"]
    fp0 = r["cands"][0]["rollouts"][0]["obs_fp"]
    feats, acts, check = [], {}, []
    for k in range(K):
        s = seed_of(tag, t, k)
        resp = client.call_endpoint("get_action", dict(req, _seed=s))
        if "error" in resp or resp.get("_seed_echo") != s or resp.get("_n_dit") not in (4, 5):
            raise RuntimeError(f"{tag} t{t} k{k}: bad response {({kk: v for kk, v in resp.items() if not kk.startswith('_feat')})}")
        feats.append({kk[1:]: v for kk, v in resp.items() if kk.startswith("_feat")})
        for kk, v in resp.items():
            if kk.startswith("action"):
                acts.setdefault(kk, []).append(np.asarray(v)[0])
        if k < len(r["cands"]):
            rec = np.load(os.path.join(data.roll(r), f"{tag}_t{t}_c{k}_m0.npz"))
            diff = max(float(np.abs(np.asarray(resp[kk])[0] - rec[kk][0]).max()) for kk in rec.files if kk.startswith("action"))
            same = r["cands"][k]["rollouts"][0]["obs_fp"] == fp0
            check.append(dict(k=k, same_req=same, exact=diff == 0.0, maxdiff=diff))
            if strict and same and diff != 0.0 and r.get("hidden") == "settled-v5":
                raise RuntimeError(f"{tag} t{t} c{k}: same request, but the chunk differs from the recorded one by {diff:.3g}")
    return {kk: np.stack([f[kk] for f in feats]) for kk in feats[0]}, {kk: np.stack(v) for kk, v in acts.items()}, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--sets", default="confirm,control,dense")
    ap.add_argument("--tasks", default=",".join(data.TASKS))
    ap.add_argument("--K", type=int, default=32, help="seeds per point in confirm / control")
    ap.add_argument("--K_dense", type=int, default=8, help="seeds per point in the other sets")
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new points (pilot)")
    ap.add_argument("--no_strict", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "feats"))
    a = ap.parse_args()
    client = BaseInferenceClient(port=a.port, timeout_ms=600000)
    todo = [(task, ds, r) for task in a.tasks.split(",") for ds in a.sets.split(",")
            for _, r in sorted(data.rows(task, ds).items())]
    todo = todo[a.worker::a.n_workers]
    print(f"worker {a.worker}/{a.n_workers}: {len(todo)} points", flush=True)
    n_new = 0
    for task, ds, r in todo:
        path = os.path.join(a.out, task, ds, f"{r['tag']}_t{r['t']}.npz")
        if os.path.exists(path):
            continue
        K = a.K if ds in ("confirm", "control") else a.K_dense
        t0 = time.time()
        feat, acts, check = extract(client, r, K, not a.no_strict)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **feat, seed=np.array([seed_of(r["tag"], r["t"], k) for k in range(K)], np.uint64),
                            **acts)
        os.replace(tmp, path)
        line = dict(task=task, ds=ds, tag=r["tag"], t=r["t"], K=K, secs=round(time.time() - t0, 2), check=check)
        with open(os.path.join(a.out, f"log_w{a.worker}.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")
        n_same = sum(c["same_req"] for c in check)
        print(f"{task} {ds} {r['tag']} t{r['t']}: K {K}, {line['secs']} s, same request {n_same}/{len(check)} "
              f"(exact {sum(c['exact'] for c in check if c['same_req'])}), max diff elsewhere "
              f"{max([c['maxdiff'] for c in check if not c['same_req']], default=0):.3g}", flush=True)
        n_new += 1
        if a.limit and n_new >= a.limit:
            break


if __name__ == "__main__":
    main()
