"""Sanity probes after the null u_t result: do the extracted features carry outcome information at all? (README.md)

u_t only measures how much the candidates' features spread; it never asks what they predict. Three checks on the same
features (z = DiT output at the action positions of the last denoising step, mean over the 16 positions; the 8 labelled
candidates; a candidate is used only if its re-sampled chunk is within 0.05 of the executed one, extract.py `maxdiff`):
  1 pipeline   Spearman(u8, raw-action variance) on coarse, within task: the features must move with the actions.
  2 state      does the candidates' mean z predict the point's V (stage success over all its rollouts)? Ridge on the
               top PCs, 5-fold cross-validation grouped by main run, per task; against a baseline from stage index and
               time alone.
  3 action     does a candidate's z, relative to the other candidates at the same point, predict its success relative
               to theirs? Ridge on centred z -> centred success, trained on every point of the other folds' main runs;
               scored at the held-out confirm / control points as the gain of the picked candidate over the mean
               candidate (unbiased: the pick does not see the test outcomes). Control: the same with the training
               outcomes shuffled within each point.
PCA and ridge are fixed (32 PCs, lambda 0.01 x n on standardized PCs), not tuned.
Feature variants (--feat; several, comma-separated):
  last         feats/ feat: the last denoising step (t = 0.75, noisy input), mean over the 16 tokens (DiVeR's)
  clean        feats_clean/ feat_clean: the clean pass over the finished chunk, mean over the tokens
  clean_first  the clean pass's first token (the first step to be executed)
  steps        the 4 denoising steps' token means, concatenated
  raw          the chunk itself (all action keys, 16 steps, flattened): no scene in it, so a linear model of it can
               only learn preferences that hold in every scene
usage: python -I probe.py [--feat last,clean,clean_first,steps,raw]   (groot python)
"""
import argparse
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "20261028_critical"))
import data  # noqa: E402

NPC, LAM, TOL, FOLDS = 32, 0.01, 0.05, 5  # LAM per sample, on standardized PCs
rng = np.random.default_rng(0)
FEAT = {  # name: (dir, labelled candidates' z from the npz)
    "last": ("feats", lambda f: f["feat"][:8].mean(1)),
    "clean": ("feats_clean", lambda f: f["feat_clean"][:8].mean(1)),
    "clean_first": ("feats_clean", lambda f: f["feat_clean"][:8, 0]),
    "steps": ("feats_clean", lambda f: f["feat_steps"][:8].reshape(8, -1)),
    "raw": ("feats_clean", lambda f: np.concatenate([f[k][:8].reshape(8, -1) for k in sorted(f.files) if k.startswith("action")], 1)),
}


def load(feat="last"):
    d, zf = FEAT[feat]
    md = {}
    for f in glob.glob(os.path.join(HERE, d, "log_w*.jsonl")):
        for r in map(json.loads, open(f)):
            md[(r["tag"], r["t"], r["ds"])] = {c["k"]: c["maxdiff"] for c in r["check"]}
    P = []
    for task in data.TASKS:
        for ds in data.DATASETS:
            for (tag, t), r in data.rows(task, ds).items():
                z = zf(np.load(os.path.join(HERE, d, task, ds, f"{tag}_t{t}.npz"))).astype(np.float64)
                y = data.outcomes(r).mean(1)
                ok = np.array([md[(tag, t, ds)][k] <= TOL for k in range(8)])
                P.append(dict(task=task, ds=ds, tag=tag, t=t, k=r["k"], z=z, y=y, ok=ok, V=float(data.outcomes(r).mean()),
                              u8=z.var(0).mean()))
    return P


def fit(X, y):
    mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:NPC].T; Z = (X - mu) @ W; sd = Z.std(0) + 1e-12; Z = Z / sd
    b = np.linalg.solve(Z.T @ Z + LAM * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - mu) @ W / sd) @ b + y.mean()


def folds(tags):
    u = sorted(set(tags)); rng.shuffle(u)
    return {tg: i % FOLDS for i, tg in enumerate(u)}


def auc(x, y):
    pos, neg = x[y], x[~y]
    d = pos[:, None] - neg[None]
    return ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size if d.size else np.nan


def check1(P):
    feats = {(r["tag"], r["t"]): r for r in map(json.loads, open(os.path.join(HERE, "..", "20261029_signals", "features.jsonl")))
             if r["ds"] == "coarse"}
    print("=== 1 pipeline: Spearman(u8, raw-action variance) on coarse")
    for task in data.TASKS:
        C = [p for p in P if p["task"] == task and p["ds"] == "coarse"]
        print(f"  {task:21s} n {len(C)}: {spearmanr([p['u8'] for p in C], [feats[(p['tag'], p['t'])]['var'] for p in C])[0]:+.2f}")


def check2(P):
    print("=== 2 state: cross-validated prediction of V from the candidates' mean z (vs stage index + time)")
    for task in data.TASKS:
        T = [p for p in P if p["task"] == task]
        fo = folds([p["tag"] for p in T]); g = np.array([fo[p["tag"]] for p in T])
        X = np.array([p["z"][p["ok"]].mean(0) if p["ok"].any() else p["z"].mean(0) for p in T])
        B = np.array([[p["t"] / 1000, *(np.arange(4) == p["k"])] for p in T], float)
        V = np.array([p["V"] for p in T]); pz, pb = np.zeros(len(T)), np.zeros(len(T))
        for f in range(FOLDS):
            tr, te = g != f, g == f
            pz[te] = fit(X[tr], V[tr])(X[te])
            Bc = np.c_[B, np.ones(len(B))]
            w = np.linalg.lstsq(Bc[tr], V[tr], rcond=None)[0]; pb[te] = Bc[te] @ w
        und = (V > 0) & (V < 1)
        print(f"  {task:21s} n {len(T)}: rho(pred, V) features {spearmanr(pz, V)[0]:+.2f}, stage+time {spearmanr(pb, V)[0]:+.2f}; "
              f"AUROC V > 0.5: features {auc(pz, V > 0.5):.2f}, stage+time {auc(pb, V > 0.5):.2f}; "
              f"undecided vs decided: features |pred - 0.5| {auc(-np.abs(pz - 0.5), und):.2f}")


def check3(P, shuffle=False):
    out = {"confirm": [], "control": []}
    for task in data.TASKS:
        T = [p for p in P if p["task"] == task]
        fo = folds([p["tag"] for p in T])
        for f in range(FOLDS):
            tr = [p for p in T if fo[p["tag"]] != f and p["ok"].sum() >= 2]
            X = np.concatenate([p["z"][p["ok"]] - p["z"][p["ok"]].mean(0) for p in tr])
            ys = []
            for p in tr:
                y = p["y"][p["ok"]] - p["y"][p["ok"]].mean()
                ys.append(rng.permutation(y) if shuffle else y)
            model = fit(X, np.concatenate(ys))
            for p in T:
                if fo[p["tag"]] == f and p["ds"] in out and p["ok"].sum() >= 2:
                    zz, yy = p["z"][p["ok"]], p["y"][p["ok"]]
                    s = model(zz - zz.mean(0))
                    out[p["ds"]].append((task, yy[np.argmax(s)] - yy.mean(), yy.max() - yy.mean()))
    return out


def report3(P):
    print("=== 3 action: gain of the candidate picked from its features over the mean candidate (held-out main runs)")
    for name, shuf in (("features", False), ("shuffled-label control", True)):
        out = check3(P, shuf)
        for ds, L in out.items():
            g = np.array([x[1] for x in L]); bs = [rng.choice(g, len(g)).mean() for _ in range(2000)]
            per = "; ".join(f"{t[:6]} {100 * np.mean([x[1] for x in L if x[0] == t]):+.0f}" for t in data.TASKS)
            print(f"  {name:22s} {ds:8s} n {len(g):2d}: {100 * g.mean():+.1f} pts [{100 * np.percentile(bs, 2.5):+.1f}, "
                  f"{100 * np.percentile(bs, 97.5):+.1f}]  ({per}); in-sample best-of-8 {100 * np.mean([x[2] for x in L]):+.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="last")
    for feat in ap.parse_args().feat.split(","):
        rng = np.random.default_rng(0)  # same folds and resamples for every variant
        P = load(feat)
        print(f"##### feature {feat}: {len(P)} points; candidates kept (re-sampled within {TOL} of the executed chunk): "
              f"{sum(p['ok'].sum() for p in P)} / {8 * len(P)}")
        if feat == "last":
            check1(P)
        check2(P)
        report3(P)
