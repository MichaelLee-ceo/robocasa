"""collect.py seeds chunk ci of candidate i, continuation m, at point (tag, t) with sha256(f"{tag}|critic|t{t}|c{i}|m{m if ci else 0}|{ci}"):
the seed does not depend on N, M or the run. A confirmation point (N=8, M=4) at the same (tag, t) as a coarse point
(N=8, M=2) or a dense point (N=8, M=1) therefore re-runs the very rollouts that located it: continuation 0 (and 1 at a
coarse point) is the same rollout again if the code and the policy server give the same numbers.
This compares, per candidate, the confirmation rollouts with the earlier ones at the same (tag, t): the outcome (done,
steps), the first chunk and the saved states. Read-only. usage: python -I check_seed_reuse.py"""
import glob, json, os
import numpy as np

DATA = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"


def load(dirs):
    out = {}
    for d in dirs:
        for p in glob.glob(os.path.join(DATA, d, "results_w*.jsonl")):
            for line in open(p):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in r:
                    out[(r["tag"], r["t"])] = (r, os.path.join(os.path.dirname(p), "roll"))
    return out


def main():
    conf = load(["run_confirm_KettleBoiling"])
    earlier = {}
    for name, dirs in [("coarse", ["run_v2/s0", "run_v2/s1", "run_v2/s2", "run_v2/s3", "run_v2/s4", "run_KettleBoiling_fix"]), ("dense", ["run_dense_KettleBoiling"])]:
        for k, v in load(dirs).items():
            earlier[k] = (name,) + v
    tot = dict(same_done=0, n=0, same_steps=0, same_chunk=0, same_states=0)
    for key in sorted(conf):
        rc, dc = conf[key]
        if key not in earlier:
            print(key, "no earlier point at the same (tag, t)")
            continue
        name, re_, de = earlier[key]
        Me = re_["M"]
        line = []
        for i in range(8):
            for m in range(Me):
                a, b = rc["cands"][i]["rollouts"][m], re_["cands"][i]["rollouts"][m]
                za, zb = np.load(f"{dc}/{key[0]}_t{key[1]}_c{i}_m{m}.npz"), np.load(f"{de}/{key[0]}_t{key[1]}_c{i}_m{m}.npz")
                ch = all(np.array_equal(za[k][0], zb[k][0]) for k in za.files if k.startswith("action"))
                n = min(len(za["states"]), len(zb["states"]))
                st = bool(np.array_equal(za["states"][:n], zb["states"][:n]))
                tot["n"] += 1; tot["same_done"] += a["done"] == b["done"]; tot["same_steps"] += a["steps"] == b["steps"]
                tot["same_chunk"] += ch; tot["same_states"] += st
                line.append(f"c{i}m{m}:{'=' if a['done'] == b['done'] else 'x'}{'=' if a['steps'] == b['steps'] else 'x'}{'C' if ch else '-'}{'S' if st else '-'}")
        vc = np.mean([[x["done"] for x in c["rollouts"]] for c in rc["cands"]])
        vreuse = np.mean([[x["done"] for x in c["rollouts"][Me:]] for c in rc["cands"]])
        print(f"{key[0]:26s} t{key[1]:4d} earlier: {name:6s} (M={Me}, {re_.get('hidden')}) vs confirm ({rc.get('hidden')}); "
              f"V confirm {vc:.2f}, V on the new continuations only {vreuse:.2f}, V earlier {np.mean([[x['done'] for x in c['rollouts']] for c in re_['cands']]):.2f}")
        print("    " + " ".join(line))
    print({k: v for k, v in tot.items()})
    print("legend: c<i>m<m>: done equal (=/x), steps equal (=/x), first chunk identical (C), saved states identical (S)")


if __name__ == "__main__":
    main()
