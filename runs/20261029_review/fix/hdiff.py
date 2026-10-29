"""Do fields that settle() leaves alone (history of earlier rollouts in the same process) change the physics? At one
boundary: settle + capture twice, with a rollout in between, diff the two captures field by field, then run the same saved
chunks from each and from a copy of the first capture with each differing field swapped in.
usage: python -I hdiff.py <dir> <task> <tag> <t> <c> <b> out.json"""
import copy, json, os, sys
import numpy as np

RV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "restore")
sys.path.insert(0, RV)
import harness as H  # noqa: E402
from imagine import set_py_state  # noqa: E402


def flat(h, pre=""):
    out = {}
    if isinstance(h, dict):
        for k, v in h.items():
            out.update(flat(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(h, list):
        for i, v in enumerate(h):
            out.update(flat(v, f"{pre}[{i}]"))
    else:
        out[pre] = h
    return out


def main():
    d, task, tag, t, c, b, outp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    row = H.find_row(d, tag, t); job = H.find_job(tag, t)
    base, kenv = H.make_env(task)
    P = H.Point(base, kenv, task, tag, t, row["k"], job["scene"], version=H.VERSION[row["hidden"]], row=row)
    saved = np.load(H.roll_path(d, tag, t, c, 0)); ss = saved["states"]

    def capture_settled():
        set_py_state(kenv, P.ps); H.set_state(kenv, ss[b]); H.settle(kenv, ss[b - 1], "v5")
        return H.hidden_state.capture(kenv)

    def run(Hh):
        st, _ = P.run(saved, ss[b], 16, Hh, chunk0=b // 16)
        return st
    Ha = capture_settled(); sa = run(Ha)
    Hb = capture_settled(); sb = run(Hb)
    fa, fb = flat(Ha), flat(Hb)
    diff = []
    for k in fa:
        x, y = fa[k], fb.get(k)
        try:
            same = np.array_equal(np.asarray(x), np.asarray(y))
        except Exception:
            same = x == y
        if not same:
            diff.append(k)
    res = dict(n_fields=len(fa), differing=diff[:60], traj_a_vs_b=float(np.abs(sa - sb).max()), per_field={})
    print("fields", len(fa), "differing", len(diff), diff[:40], "| run a vs b max", res["traj_a_vs_b"], flush=True)
    # which differing top-level piece matters: run from Ha with one piece taken from Hb
    pieces = sorted({k.split(".")[0] if not k.startswith("robots") else ".".join(k.split(".")[:3]) for k in diff})
    for pc in pieces:
        Hc = copy.deepcopy(Ha)
        if pc.startswith("robots"):
            i = int(pc.split("[")[1].split("]")[0]); grp, name = pc.split(".")[1], pc.split(".")[2]
            Hc["robots"][i][grp][name] = copy.deepcopy(Hb["robots"][i][grp][name])
        else:
            Hc[pc] = copy.deepcopy(Hb[pc])
        sc = run(Hc)
        res["per_field"][pc] = float(np.abs(sc - sa).max())
        print(pc, "-> traj diff vs a:", res["per_field"][pc], flush=True)
    sa2 = run(Ha)
    res["a_again_vs_a"] = float(np.abs(sa2 - sa).max())
    print("a run again vs a:", res["a_again_vs_a"])
    json.dump(res, open(outp, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
