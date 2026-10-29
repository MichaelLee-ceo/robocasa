"""Pick chunk boundaries for ramp_restore.py (state error caused by each gripper-ramp rule): per situation, boundaries
where the rules disagree (or a typical one), preferring early boundaries (short continuous replays) and rollouts that
already contribute. Writes cases_ramp_<task>.json.
usage: python -I select_ramp_cases.py"""
import json, os, random, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_eval import rules, situations  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WANT = {  # situation -> {task: n}
    "free, 75.2-78.5 mm (creep band)": {"KettleBoiling": 1, "SteamInMicrowave": 1, "PackIdenticalLunches": 1},
    "holding, wide grasp >= 60 mm": {"SteamInMicrowave": 1, "PackIdenticalLunches": 1},
    "holding, 20-60 mm": {"SteamInMicrowave": 1},
    "holding, < 20 mm": {"KettleBoiling": 1},
    "touching an object, not held": {"KettleBoiling": 1, "PackIdenticalLunches": 1},
    "touching a fixture, mid gap": {"KettleBoiling": 1, "SteamInMicrowave": 2},
    "mid-ramp, releasing (last cmd open)": {"KettleBoiling": 1, "PackIdenticalLunches": 1},
    "mid-ramp, grasping (last cmd close)": {"SteamInMicrowave": 1, "PackIdenticalLunches": 1},
    "mid-ramp, steady mid gap": {"KettleBoiling": 1, "SteamInMicrowave": 1},
}


def main():
    Z = np.load(f"{HERE}/out/grip_table.npz", allow_pickle=True)
    T = {k: Z[k] for k in Z.files}
    R = rules(T)
    lab = situations(T)
    names = ["a_v3", "b_v4", "c_v5", "d_val", "f_hyb"]
    out = defaultdict(lambda: defaultdict(lambda: dict(boundaries=[], labels=[])))
    rnd = random.Random(1)
    for sit, per in WANT.items():
        for task, n in per.items():
            ti = list(T["tasks"]).index(task)
            idx = np.nonzero((lab == sit) & (T["task"] == ti))[0]
            if not len(idx):
                print("no match", sit, task); continue
            keys = [T["key"][i].split("|") for i in idx]
            b = np.array([int(k[4]) for k in keys])
            dis = np.array([len({float(R[r][i]) for r in names}) > 1 for i in idx])
            # prefer disagreement, then early boundaries
            order = sorted(range(len(idx)), key=lambda j: (not dis[j], b[j] > 160, rnd.random()))
            for j in order[:n]:
                i = idx[j]; d, tag, t, c, bb = T["key"][i].split("|")
                cs = out[task][(d, tag, int(t), int(c))]
                cs["boundaries"].append(int(bb))
                cs["labels"].append(f"b{bb}: {sit}; c_true {T['c'][i]:+.1f}; rules " + ",".join(f"{r}={R[r][i]:+.1f}" for r in names))
    for task, cases in out.items():
        lst = [dict(dir=k[0], task=task, tag=k[1], t=k[2], c=k[3], m=0, boundaries=v["boundaries"], label=" | ".join(v["labels"]))
               for k, v in cases.items()]
        lst.sort(key=lambda x: max(x["boundaries"]))
        json.dump(lst, open(f"{HERE}/cases_ramp_{task}.json", "w"), indent=1)
        print(task, len(lst), "rollouts", sum(len(x["boundaries"]) for x in lst), "boundaries")
        for x in lst:
            print("   ", x["dir"], x["tag"], x["t"], x["c"], x["label"][:300])


if __name__ == "__main__":
    main()
