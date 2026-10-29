"""Checks settle_candidate.gripper_closing in the real env against the offline rule g_final (grip_eval_geo.py) and the
true ramp, on a stratified sample of scanned boundaries of one task (fixture contact with a mid gap, held objects, the
creep band, and random others). No stepping: set_state on the saved boundary state only.
usage: python -I validate_candidate.py <task> <n_per_stratum> out.json"""
import json, os, random, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
from settle_candidate import gripper_closing  # noqa: E402
from grip_eval import rules, situations  # noqa: E402


def main():
    task, n, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    Z = np.load(f"{H.HERE}/out/grip_table.npz", allow_pickle=True)
    T = {k: Z[k] for k in Z.files}
    geo = json.load(open(f"{H.HERE}/out/grip_geo.json"))
    R = rules(T); lab = situations(T)
    g0 = T["hist"][:, -1]
    nin = np.array([geo.get(k, (0, 0))[0] for k in T["key"]]); nout = np.array([geo.get(k, (0, 0))[1] for k in T["key"]])
    side = np.where(nin > nout, 1.0, np.where(nout > nin, -1.0, 0.0))
    gf = R["a_v3"].copy()
    use = T["fix"] & ~T["obj"] & ~T["held"] & (g0 >= 0.005) & (g0 < 0.07517) & (side != 0)
    gf[use] = side[use]
    gf[T["held"]] = np.where(g0[T["held"]] < 0.0785, 1.0, -1.0)
    ti = list(T["tasks"]).index(task)
    rnd = random.Random(0)
    strata = {"fixture mid gap": lab == "touching a fixture, mid gap", "held": T["held"],
              "creep band": lab == "free, 75.2-78.5 mm (creep band)", "other": np.ones(len(gf), bool)}
    picks = []
    for sname, m in strata.items():
        idx = list(np.nonzero(m & (T["task"] == ti) & (np.abs(T["c"]) > 0.999))[0])
        rnd.shuffle(idx)
        picks += [(sname, i) for i in idx[:n]]
    by_tag = defaultdict(list)
    for sname, i in picks:
        by_tag[T["key"][i].split("|")[1]].append((sname, i))
    base, kenv = H.make_env(task)
    res = Counter(); rows = []
    for tag, lst in by_tag.items():
        job = H.find_job(tag, int(T["key"][lst[0][1]].split("|")[2]))
        H.load_scene(kenv, job["scene"])
        for sname, i in lst:
            d, _, t, c, b = T["key"][i].split("|")
            S = np.load(H.roll_path(d, tag, int(t), int(c), 0))["states"]
            H.set_state(kenv, S[int(b)])
            v = 1.0 if gripper_closing(kenv, kenv.robots[0], "right", S[int(b) - 1]) else -1.0
            agree = v == gf[i]
            right = v == np.sign(T["c"][i])
            res[(sname, "agree_offline")] += agree; res[(sname, "right")] += right; res[(sname, "n")] += 1
            rows.append(dict(key=T["key"][i], stratum=sname, env=v, offline=float(gf[i]), c_true=float(T["c"][i])))
        print(tag, dict(res), flush=True)
    json.dump(dict(summary={f"{k[0]}|{k[1]}": int(v) for k, v in res.items()}, rows=rows), open(outp, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
