"""Are the continuations of a candidate exchangeable, as every estimator assumes? Continuation m >= 1 replays the stored
first chunk, and m only changes the later chunks' noise seeds, so P(done) must not depend on m (nor on the candidate's
position c, since c only changes the seeds). The rollouts of a point run one after another in one env, so a dependence on
m or c would point to state carried over between rollouts. Per dir: done rate by m and by c, with a paired sign test
over points (m = 0 vs the rest). Read-only. usage: python -I check_exchangeable.py"""
import glob, json, math, os
import numpy as np

DATA = "/home/michaellee/mclee/robocasa/runs/20261026_critic_data"
DIRS = ["run_v2/s*", "run_SteamInMicrowave_v2", "run_PackIdenticalLunches", "run_PackIdenticalLunches_fix", "run_SteamInMicrowave_fix",
        "run_SteamInMicrowave_fix3", "run_PackIdenticalLunches_fix3", "run_confirm_KettleBoiling", "run_dense_KettleBoiling",
        "run_dense_SteamInMicrowave", "run_dense_PackIdenticalLunches", "run_dense2_SteamInMicrowave", "run_dense2_PackIdenticalLunches"]


def sign_p(k, n):
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)


def main():
    for d in DIRS:
        Y = []
        for p in glob.glob(os.path.join(DATA, d, "results_w*.jsonl")):
            for line in open(p):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in r:
                    Y.append(np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float))
        Y = np.stack(Y)  # points x N x M
        n, N, M = Y.shape
        byc = Y.mean((0, 2))
        s = f"{d:32s} points {n:3d}  done by c: " + " ".join(f"{v:.2f}" for v in byc)
        if M > 1:
            bym = Y.mean((0, 1))
            dm = Y[:, :, 0].mean(1) - Y[:, :, 1:].mean((1, 2))
            pos, nz = int((dm > 0).sum()), int((dm != 0).sum())
            s += f"  | by m: " + " ".join(f"{v:.3f}" for v in bym) + f"  m0 - rest: mean {dm.mean():+.3f}, higher at {pos}/{nz} points, sign p {sign_p(pos, nz):.2f}"
        dc = Y[:, 0].mean(1) - Y[:, 1:].mean((1, 2))
        pos, nz = int((dc > 0).sum()), int((dc != 0).sum())
        s += f"  | c0 - rest {dc.mean():+.3f}, sign p {sign_p(pos, nz):.2f}"
        print(s)


if __name__ == "__main__":
    main()
