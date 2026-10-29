"""Checks hidden_state.gripper_ramp (settled-v5) in the real env against the same rule computed offline from the restore
review's ground-truth table (out/grip_table.npz: true ramp c at chunk boundaries of saved rollouts) and against the true
ramp, on a stratified sample of one task's boundaries. No stepping: set_state on the saved boundary state only.
usage: python -I validate_v5.py <task> <n_per_stratum> out.json"""
import json, os, random, sys
from collections import Counter, defaultdict
import numpy as np

RV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "restore")
sys.path.insert(0, RV)
import harness as H  # noqa: E402
from grip_eval import situations  # noqa: E402


def offline_v5(T, geo):
    """gripper_ramp's rule, vectorised over the table (contacts as grip_table / grip_geo recorded them)."""
    h = T["hist"]; g0, g1 = h[:, -1], h[:, -2]; v1 = g0 - g1
    held, obj, fix = T["held"], T["obj"], T["fix"]
    nin = np.array([geo.get(k, (0, 0))[0] for k in T["key"]]); nout = np.array([geo.get(k, (0, 0))[1] for k in T["key"]])
    old = np.where((g0 >= H.hidden_state.OPEN_OLD) | (v1 > 0.0005), -1.0, 1.0)
    servo = np.clip(np.round((1 - (g0 + 2 * v1) / 0.04) / 0.2) * 0.2, -1, 1)
    side = np.where(nin > nout, 1.0, np.where(nout > nin, -1.0, 0.0))
    r = servo.copy()
    fix_still = fix & ~obj & ~held & (np.abs(v1) <= 0.0005)
    r[fix_still] = old[fix_still]
    cs = fix_still & (g0 >= 0.005) & (g0 < H.hidden_state.OPEN_OLD) & (side != 0)
    r[cs] = side[cs]
    both = fix & obj & ~held
    r[both] = old[both]
    r[held] = np.where(g0[held] < 0.0785, 1.0, -1.0)
    return r


def main():
    task, n, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    Z = np.load(f"{RV}/out/grip_table.npz", allow_pickle=True)
    T = {k: Z[k] for k in Z.files}
    geo = json.load(open(f"{RV}/out/grip_geo.json"))
    lab = situations(T)
    off = offline_v5(T, geo)
    ti = list(T["tasks"]).index(task)
    rnd = random.Random(1)
    mid = np.abs(T["c"]) < 0.999
    strata = {"mid-ramp": mid, "free mid gap": lab == "free, mid gap 5-75 mm", "fixture": T["fix"] & ~T["obj"] & ~T["held"],
              "held": T["held"], "object not held": T["obj"] & ~T["held"], "creep band": lab == "free, 75.2-78.5 mm (creep band)",
              "other": np.ones(len(off), bool)}
    picks = []
    for sname, m in strata.items():
        idx = list(np.nonzero(m & (T["task"] == ti))[0])
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
            v = H.hidden_state.gripper_ramp(kenv, kenv.robots[0], "right", S[int(b) - 1])
            err = abs(v - T["c"][i]) / 0.2
            res[(sname, "n")] += 1; res[(sname, "agree_offline")] += int(abs(v - off[i]) < 1e-9)
            res[(sname, "err_steps_x100")] += int(round(100 * err)); res[(sname, "err_ge2")] += int(err >= 1.999)
            rows.append(dict(key=T["key"][i], stratum=sname, env=v, offline=float(off[i]), c_true=float(T["c"][i])))
        print(tag, dict(res), flush=True)
    summ = {}
    for sname in strata:
        nn = res[(sname, "n")]
        if nn:
            summ[sname] = dict(n=nn, agree_offline=res[(sname, "agree_offline")], mean_err_steps=round(res[(sname, "err_steps_x100")] / 100 / nn, 2),
                               err_ge2=res[(sname, "err_ge2")])
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, rows=rows), open(outp, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
