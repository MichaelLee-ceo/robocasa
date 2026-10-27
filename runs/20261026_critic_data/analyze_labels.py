"""Label statistics of a collection run: how often the 8 candidates of a decision point disagree, and how much of the
outcome (stage done within L) the first chunk decides, from the two continuations of every candidate.
- mix: per decision point, all candidates finish / none finish / mixed (counting both continuations).
- same vs cross: P(two continuations agree) for the same candidate vs for two different candidates of the same point.
  If the first chunk did not matter, both would be equal.
- ICC: share of the outcome variance between candidates (one-way ANOVA estimator with 2 continuations each), pooled
  over decision points.
- pick-by-m0: choose a candidate whose continuation 0 finished (perfect judge of one sample), score its continuation 1;
  compare with the mean of continuation 1 over all candidates (= a random pick). Unbiased: selection and evaluation use
  different noise.
usage: python analyze_labels.py [run dir=run]"""
import glob, json, sys
from collections import defaultdict
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "run"
rows = [r for p in glob.glob(f"{D}/results_w*.jsonl") for r in map(json.loads, open(p)) if "error" not in r]
by = defaultdict(list)
for r in rows:
    by["all"].append(r); by[f"S{r['k']}"].append(r)
print(f"{D}: {len(rows)} decision points")
print(f"{'':5s} {'n':>4s} {'all':>4s} {'none':>5s} {'mixed':>6s} {'P(done)':>8s} {'same':>6s} {'cross':>6s} {'ICC':>5s} "
      f"{'random':>7s} {'pick':>6s} {'n_pick':>6s}")
for key in ["all"] + sorted(k for k in by if k != "all"):
    rs = by[key]
    Y = [np.array([[x["done"] for x in c["rollouts"]] for c in r["cands"]], float) for r in rs]  # N x M
    mix = [("all" if y.min() == 1 else "none" if y.max() == 0 else "mixed") for y in Y]
    same = np.mean([float(y[i, 0] == y[i, 1]) for y in Y for i in range(len(y))])
    cross = np.mean([float(y[i, a] == y[j, b]) for y in Y for i in range(len(y)) for j in range(len(y)) if i != j
                     for a in range(2) for b in range(2)])
    sw = np.mean([((y[:, 0] - y[:, 1]) ** 2 / 2).mean() for y in Y])          # within-candidate variance
    sb = np.mean([max(y.mean(1).var(ddof=1) - ((y[:, 0] - y[:, 1]) ** 2 / 2).mean() / 2, 0) for y in Y])  # between
    icc = sb / (sb + sw) if sb + sw else float("nan")
    rand = np.mean([y[:, 1].mean() for y in Y])
    picks = [y[y[:, 0] == 1, 1].mean() for y in Y if y[:, 0].any()]
    rand_p = np.mean([y[:, 1].mean() for y in Y if y[:, 0].any()])
    print(f"{key:5s} {len(rs):4d} {mix.count('all'):4d} {mix.count('none'):5d} {mix.count('mixed'):6d} "
          f"{np.mean([y.mean() for y in Y]):8.2f} {same:6.2f} {cross:6.2f} {icc:5.2f} {rand_p:7.2f} {np.mean(picks):6.2f} {len(picks):6d}")
print("random/pick: P(continuation 1 finishes) for a random candidate vs a candidate whose continuation 0 finished,"
      " over points where some continuation 0 finished")
